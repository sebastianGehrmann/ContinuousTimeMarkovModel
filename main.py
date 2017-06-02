import argparse
from pickle import load

import numpy as np
import theano
import theano.tensor as TT
from pymc3 import Model, sample, Dirichlet, Potential, Beta, NUTS, Constant
from helpers.forwardS import ForwardS
from helpers.forwardX import ForwardX

from scipy.special import logit
from theano.tensor import as_tensor_variable

from helpers.distributions import Beta_with_anchors
from helpers.distributions import Claims
from helpers.distributions import Comorbidities
from helpers.distributions import DiscreteObsMJP
from helpers.distributions import DiscreteObsMJP_unif_prior


from helpers.theanomod import DES_diff

sampleVars = ['Q', 'pi', 'S', 'B0', 'B', 'X', 'Z', 'L']

# Set up inputs
parser = argparse.ArgumentParser(description='Run disease progression model.')
parser.add_argument('-d', '--dir', action='store', default='data/converted/', type=str, dest='datadir',
                    help='directory with pickled initial model parameters and observations')
parser.add_argument('-n', '--sampleNum', action='store', default=1001, type=int, dest='sampleNum',
                    help='number of samples to run')
parser.add_argument('-t', '--truncN', action='store', default=None, type=int,
                    help='number of people to truncate sample to')
parser.add_argument('-c', '--const', action='store', default=[], nargs='+', type=str, choices=sampleVars,
                    dest='constantVars',
                    help='list of variables to hold constant during sampling')
parser.add_argument('--seed', action='store', default=111, type=int, dest='random_seed',
                    help='random seed for sampling')
parser.add_argument('-p', '--profile', action='store_true', dest='profile',
                    help='turns on theano profiler')
parser.add_argument('-P', '--hide-progressbar', action='store_false', dest='progressbar',
                    help='hides progress bar in sample')
# parser.add_argument('outfile', nargs='?', type=argparse.FileType('w'), default=sys.stdout)
args = parser.parse_args()

if args.profile:
    theano.config.profile = True

datadir = args.datadir

# Utility function to load preprocessed files
def read_pkl(fname):
    print fname
    with open(datadir + fname + ".pkl", 'rb') as f:
        d =  load(f)
        print d
        return d

print "loading files"

# D = #Clinical Findings, N = #Patients, M = #States, K = #Comorbidities
pi_start = read_pkl('pi') # State distribution,              Size M
Q_start = read_pkl('Q')   # Transition Matrix,               Size M x M
S_start = read_pkl("S")   # initial State for each patient,  Size N
B_start = read_pkl("B")   # Onset Prob for Comorbs,          Size K x M
B0_start = read_pkl("B0") # Onset Prob for Comorb at time 0, Size K x M
X_start = read_pkl("X")   # Comorbidities for each patient,  Size P x K
Z_start = read_pkl("Z")   # Onset Prob for Clinical Codes,   Size K x D
L_start = read_pkl("L")   # Leak Prob for Clinical Codes,    Size D
# obs_jumps[o]: number of time periods that have passed between observation o and o-1, Size D
obs_jumps = read_pkl("obs_jumps")
# T[n]: total number of observations of patient n, Size N
T = read_pkl("T")
# O[o,:]: claims numbers present at observation o padded by -1's, Size (D, maxFindings)
O = read_pkl("O")
# List of Anchors
anchors = read_pkl("anchors")

print "Truncate to max", args.truncN,"#people based on args.truncN"
if args.truncN is not None:
    T = T[:args.truncN]
    nObs = T.sum()
    S_start = S_start[0:nObs]
    obs_jumps = obs_jumps[0:nObs]
    X_start = X_start[0:nObs]
    O = O[0:nObs]

nObs = S_start.shape[0]  # Number of observations
N = T.shape[0]           # Number of patients
M = pi_start.shape[0]    # Number of hidden states
K = Z_start.shape[0]     # Number of comorbidities
D = Z_start.shape[1]     # Number of claims
Dmax = O.shape[1]        # Maximum number of claims that can occur at once

print "Apply Mask on Anchors"

# Assumption that icd9-anchors are exclusive
# For all the fixed assignments to comorbs, set all other probabilities in Z to 0
mask = np.ones((K, D))
for anchor in anchors:
    for hold in anchor[1]:
        mask[:, hold] = 0
        mask[anchor[0], hold] = 1
Z_start = Z_start[mask.nonzero()]


print "Transform initial params to log probs"
# Transform the initial parameters to log probs
# Q_raw is just forward probability from 1 to 2, 2 to 3 etc.
Q_raw = []
for i in range(Q_start.shape[0] - 1):
    Q_raw.append(Q_start[i, i + 1])
Q_raw_log = logit(np.asarray(Q_raw))

B_lo = logit(B_start)
B0_lo = logit(B0_start)
Z_lo = logit(Z_start)
L_lo = logit(L_start)

start = {'Q_ratematrixoneway_': Q_raw_log, 'B_logodds_': B_lo, 'B0_logodds_': B0_lo, 'S': S_start, 'X': X_start,
         'Z_anchoredbeta_': Z_lo, 'L_logodds_': L_lo}

print "Initialize the model"

# Initialize the PyMC model, set variables with priors
model = Model()
with model:
    # pi[m]: probability of starting in disease state m
    pi = Dirichlet('pi', a=as_tensor_variable(pi_start.copy()), shape=M)
    # opt. constraint on pi - add a penalty that makes sure that every value in pi is > .001
    pi_min_potential = Potential('pi_min_potential', TT.switch(TT.min(pi) < .001, -np.inf, 0))


    # exp(t*Q)[m,m']: probability of transitioning from disease state m to m' after a period of time t
    Q = DiscreteObsMJP_unif_prior('Q', M=M, lower=0.0, upper=1.0, shape=(M, M))

    # Define probability distribution, has potential to compute C for E-step and time jumps
    S = DiscreteObsMJP('S', pi=pi, Q=Q, M=M, nObs=nObs, observed_jumps=obs_jumps, T=T, shape=(nObs))

    # Comorbidity onset probabilities B, and Z and L Beta priors (p.4, because conjugate for bernoulli)
    B0 = Beta('B0', alpha=1., beta=1., shape=(K, M))
    # constrain B0 to be monotonous
    B0_monotonicity_constraint = Potential('B0_monotonicity_constraint',
                                           TT.switch(TT.min(DES_diff(B0)) < 0., 100.0 * TT.min(DES_diff(B0)), 0))

    B = Beta('B', alpha=1., beta=1., shape=(K, M))

    X = Comorbidities('X', S=S, B0=B0, B=B, T=T, shape=(nObs, K))
    # Extension of Beta here because anchors have to be fixed, keeps mask on Beta
    Z = Beta_with_anchors('Z', anchors=anchors, K=K, D=D, alpha=0.1, beta=1., shape=(K, D))
    L = Beta('L', alpha=1., beta=1., shape=D)
    O_obs = Claims('O_obs', X=X, Z=Z, L=L, T=T, D=D, O_input=O, shape=(nObs, Dmax), observed=O)

print "model initialized"


with model:
    steps = []

    if 'pi' in args.constantVars:
        steps.append(Constant(vars=[pi]))
    else:
        steps.append(NUTS(vars=[pi]))
    if 'Q' in args.constantVars:
        steps.append(Constant(vars=[Q]))
    else:
        steps.append(NUTS(vars=[Q], scaling=np.ones(M - 1, dtype=float) * 10.))
    print "S"
    if 'S' in args.constantVars:
        steps.append(Constant(vars=[S]))
    else:
        steps.append(ForwardS(vars=[S], nObs=nObs, T=T, N=N, observed_jumps=obs_jumps))
    print "B"
    if 'B0' in args.constantVars:
        steps.append(Constant(vars=[B0]))
        if 'B' in args.constantVars:
            steps.append(Constant(vars=[B]))
        else:
            steps.append(NUTS(vars=[B]))
    elif 'B' in args.constantVars:
        steps.append(NUTS(vars=[B0]))
        steps.append(Constant(vars=[B]))
    else:
        steps.append(NUTS(vars=[B0, B]))
    print "X"
    if 'X' in args.constantVars:
        steps.append(Constant(vars=[X]))
    else:
        steps.append(ForwardX(vars=[X], N=N, T=T, K=K, D=D, Dd=Dmax, O=O, nObs=nObs))
    print "Z"
    if 'Z' in args.constantVars:
        steps.append(Constant(vars=[Z]))
    else:
        # import pdb; pdb.set_trace()
        steps.append(NUTS(vars=[Z]))
    print "L"
    if 'L' in args.constantVars:
        steps.append(Constant(vars=[L]))
    else:
        steps.append(NUTS(vars=[L], scaling=np.ones(D)))
    print "sample"
    trace = sample(args.sampleNum, steps, start=start, random_seed=args.random_seed, progressbar=args.progressbar)

print "getting traces now"
pi = trace[pi]
Q = trace[Q]
S = trace[S]
B0 = trace[B0]
B = trace[B]
X = trace[X]
Z = trace[Z]
L = trace[L]

from pickle import dump
with open('trace.pkl','wb') as file:
  dump(trace,file)

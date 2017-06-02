import argparse
import numpy as np
import os
import pandas as pd
from pickle import dump
import re
from scipy.linalg import expm

pd.options.mode.chained_assignment = None

# Set up inputs
parser = argparse.ArgumentParser(description='Convert claims into inputs for the Sontag disease progression model.')
parser.add_argument(action='store', default='data/param_init_small/claimsdata.csv', type=str, dest='claimsfile',
                    help='claims csv file to read in')
parser.add_argument('-o', '--outdir', action='store', default='data/converted/', type=str, dest='outdir',
                    help='directory to output data')
parser.add_argument('-p', '--paramdir', action='store', default=None, type=str, dest='paramdir',
                    help='directory to grab parameter initializations from')
parser.add_argument('-a', '--anchorsdir', action='store', default=None, type=str, dest='anchorsdir',
                    help='directory to grab anchors from if not specifying paramdir')
parser.add_argument('-c', '--maxclaims', action='store', default=None, type=int, dest='maxclaims',
                    help='Only use the top c occurring codes')
parser.add_argument('-s', '--minsteps', action='store', default=3, type=int, dest='minsteps',
                    help='minimum number of active time periods')
parser.add_argument('--seed', action='store', default=111, type=int, dest='randomseed',
                    help='random seed for sampling')
parser.add_argument('--kcomorbid', action='store', default=4, type=int, dest='K',
                    help='specify K if not specifying paramdir')
parser.add_argument('--mstates', action='store', default=4, type=int, dest='M',
                    help='specify M if not specifying paramdir')

args = parser.parse_args()


def loadAnchors(dataDirectory, icd9Map):
    """
    Loads Anchor File
    Returns Names of Comorbidities and List of Anchors for each
    """
    comorbidityNames = []
    anchors = []
    # Load all anchors
    print icd9Map.keys()
    with open(dataDirectory + '/anchor_icd9.csv') as anchorFile:
        for i, line in enumerate(anchorFile):
            text = line.strip().split(',')
            comorbidityNames.append(text[0])
            comorbAnchors = []
            for codeStr in text[1:]:
                for key in icd9Map.keys():
                    l = re.search(codeStr, key)
                    if l is not None:
                        print "Found ICD code", key
                        comorbAnchors.append(icd9Map[key])
            anchors.append((i, comorbAnchors))
    anchorFile.close()
    return anchors, comorbidityNames


def writeDict(outfile, d):
    """
    Takes name of a file and a dictionary and stores it
    """
    out = open(outfile, "w")
    items = [(v, k) for k, v in d.iteritems()]
    items.sort()
    for v, k in items:
        print >> out, k, v
    out.close()

# Set Random Seed
np.random.seed(args.randomseed)

# Load Claims
claimsDF = pd.read_csv(args.claimsfile)
print claimsDF.head()

# Drop all codes that are not in the top "args.maxclaims" occurring
if args.maxclaims is not None:
    fid = claimsDF.Code.value_counts()[0:args.maxclaims].index.values
    claimsDF.Code = claimsDF.Code.apply(lambda x: x if x in fid else np.nan)
    claimsDF.dropna(inplace=True)
else:
    fid = claimsDF.Code.unique()
D = len(fid)

# Map the codes to numbers
fidDict = {}
for i, icd9 in enumerate(fid):
    fidDict[str(icd9)] = i

# Save dict to file for later analysis
writeDict(os.path.join(args.outdir, "fid.dict"), fidDict)
# Convert all Codes to the numbers
claimsDF.loc[:, 'Code'] = claimsDF.Code.apply(lambda x: fidDict[str(x)])


print claimsDF.head()

# Drop Patients with < args.minsteps tiem periods of data
finalClaims = claimsDF.groupby(['PID'], as_index=False).apply(
    lambda x: x if x.time_delta.nunique() >= args.minsteps else None).reset_index(drop=True)

Dmax = finalClaims.groupby(['PID', 'time_delta']).count().max()[0]
print "Maximum Number of Time Periods with data:", Dmax

# T: Number of Unique Observations per Patient
T = finalClaims.groupby(['PID']).time_delta.nunique().values
# Total Number of Observations
nObs = T.sum()
print "Total Number of Observations", nObs
N = len(T)
print "Number Patients:", N

# The indices of where patients start
zeroIndices = np.roll(T.cumsum(), 1)
zeroIndices[0] = 0

# Initialize O as matrix of Time periods x Number Observations, -1 = no obs
O = np.ones((nObs, Dmax), dtype=int) * -1
# Initialize obs_jumps of size Number Observations - measures the time difference between obs
obs_jumps = np.zeros((nObs), dtype=int)

# Fill observation matrix for each patient and the time deltas
counter = 0
prevTime = 0
for group in finalClaims.groupby(['PID', 'time_delta']):
    for i, val in enumerate(group[1].Code):
        O[counter, i] = val
    curTime = group[1].time_delta.values[0]
    obs_jumps[counter] = curTime - prevTime
    prevTime = curTime
    counter += 1
# Set jumps to 0 whereever a new patient starts
obs_jumps[zeroIndices] = 0


# Initialize Model Matrices
if args.paramdir is not None:
    dataDirectory = args.paramdir
    B = np.loadtxt(dataDirectory + '/B.txt')
    B0 = np.loadtxt(dataDirectory + '/B0.txt')
    L = np.loadtxt(dataDirectory + '/L.txt')
    pi = np.loadtxt(dataDirectory + '/pi.txt')
    Q = np.loadtxt(dataDirectory + '/Q.txt')
    Z = np.loadtxt(dataDirectory + '/Z.txt')
    anchors, comorbidityNames = loadAnchors(dataDirectory, fidDict)
    M = pi.shape[0]
    K = Z.shape[0]
    print "Used files to initialize Matrices"
else:
    print "Generate initializations for matrices"
    # Number Comorbidities
    K = args.K
    # Number States
    M = args.M
    # Leak Probabilities
    L = np.random.rand(D) * 0.3
    # Activation Probability for Each Observation for each Comorbidity
    Z = np.random.rand(K, D)
    # Comorbidity Onset Probability for turning off
    B = np.random.rand(K, M)
    # Comorbidity Onset Probability for turning on
    B0 = np.random.rand(K, M)
    B0.sort(axis=1)
    # Initial State Probability
    pi = np.random.rand(M) * (1 - M * 0.001) + 0.001 * M
    pi = pi / pi.sum()
    pi[::-1].sort()
    # Transition Probabilities between states
    Qvals = np.random.rand(M - 1)
    Q = np.zeros((M, M))
    for i, val in enumerate(Qvals):
        Q[i, i + 1] = val
        Q[i, i] = -val
    # Initialize Anchors
    if args.anchorsdir is not None:
        anchors, comorbidityNames = loadAnchors(args.anchorsdir)
    else:
        anchors = []
        comorbidityNames = []


# Transition probability between states given a time delta
jumpInd = {}
transMat = []
for i, jump in enumerate(np.unique(obs_jumps)[1:]):
    jumpInd[jump] = i
    # from paper: Matrix Exponential delta * Q (p.3)
    transMat.append(expm(jump * Q))



# Generate S (States) from parameters (p.3)
S = np.zeros(nObs, dtype=np.int32)
# Initial States sampled from pi
S[zeroIndices] = np.random.choice(np.arange(M), size=(N), p=pi)
# for all patients get all observations and sample progression based on initialized model and state
for n in range(N):
    n0 = zeroIndices[n]
    for t in range(1, T[n]):
        print n0, t
        print obs_jumps[n0+t]
        print jumpInd[obs_jumps[n0 + t]]
        print transMat[jumpInd[obs_jumps[n0 + t]]]
        print transMat[jumpInd[obs_jumps[n0 + t]]][S[n0 + t - 1]]
        S[n0 + t] = np.random.choice(np.arange(M), p=transMat[jumpInd[obs_jumps[n0 + t]]][S[n0 + t - 1]])

# Generate X (Comorbidities) from parameters
X = np.zeros((nObs, K))
# Activate Initial based on binomial sample of B0 (Comorbidity initialization)
X[zeroIndices] = np.random.binomial(n=1, p=B0[:, S[zeroIndices]].T)
# For all Comorbidities for all patients
for k in range(K):
    for n in range(N):
        # If Patient is in same visit as before, keep activated
        n0 = zeroIndices[n]
        if X[n0, k] == 1:
            X[zeroIndices[n]:(zeroIndices[n] + T[n]), k] = 1
        # Otherwise sample from the transition matrix B given state S
        else:
            # Get the change in states for the current patient (i.e. 0 when no change)
            changed = np.diff(S[zeroIndices[n]:(zeroIndices[n] + T[n])])
            for t in range(1, T[n]):
                # Activate Comorbidities based on activations in B
                if changed[t - 1] == 1 and np.random.rand() < B[k, S[n0 + t]]:
                    X[(n0 + t):(zeroIndices[n] + T[n]), k] = 1
                    break
# Change type
X = X.astype(np.int8)
#
# Write pickled files
variables = [Q, pi, S, T, obs_jumps, B0, B, X, Z, L, O, anchors, comorbidityNames]
names = ['Q', 'pi', 'S', 'T', 'obs_jumps', 'B0', 'B', 'X', 'Z', 'L', 'O', 'anchors', 'comorbidityNames']
if not os.path.isdir(args.outdir):
    os.mkdir(args.outdir)
for var, name in zip(variables, names):
    outfile = open(args.outdir + '/' + name + '.pkl', 'wb')
    dump(var, outfile)
    outfile.close()

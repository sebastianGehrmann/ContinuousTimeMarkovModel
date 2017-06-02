import numpy as np
import theano.tensor as TT

from pymc3 import Continuous
from pymc3.distributions.dist_math import bound, logpow
from pymc3.distributions.special import gammaln
from pymc3.distributions.discrete import Categorical, Binomial
from transforms import rate_matrix_one_way, rate_matrix, anchored_betas
#from theano.tensor.nlinalg import eig, matrix_inverse
from theano.compile.sharedvalue import shared
import theano.tensor.slinalg
from theano.tensor.extra_ops import bincount

import profilingUtil


class DiscreteObsMJP_unif_prior(Continuous):
    def __init__(self, M, lower=0, upper=100, *args, **kwargs):
        self.lower = lower
        self.upper = upper
        super(DiscreteObsMJP_unif_prior, self).__init__(transform=rate_matrix_one_way(lower=lower, upper=upper), *args,
                                                               **kwargs)
        Q = np.ones((M, M), np.float64) - .5
        self.mode = Q

    def logp(self, value):
        return TT.as_tensor_variable(0.0)


class Beta_with_anchors(Continuous):
    def __init__(self, anchors, K, D, alpha=1.0, beta=1.0, *args, **kwargs):
        self.alpha = shared(alpha)
        self.beta = shared(beta)
        # mask contains zeros for elements fixed at 1E-6
        mask = np.ones((K, D))
        # repeat masking, so that comorb stays fixed
        for anchor in anchors:
            # mask[:,anchor[1]] = 0
            for hold in anchor[1]:
                mask[:, hold] = 0
                mask[anchor[0], hold] = 1
        self.mask = TT.as_tensor_variable(mask)

        super(Beta_with_anchors, self).__init__(
            transform=anchored_betas(mask=self.mask, K=K, D=D, alpha=alpha, beta=beta), *args, **kwargs)

        self.mean = TT.ones_like(self.mask) * 1E-6
        self.mean = TT.set_subtensor(self.mean[self.mask.nonzero()], (alpha / (alpha + beta)))

    def logp(self, Z):
        alpha = self.alpha
        beta = self.beta

        Zanchored = Z[self.mask.nonzero()]

        logp = bound(
            gammaln(alpha + beta) - gammaln(alpha) - gammaln(beta) +
            logpow(
                Zanchored, alpha - 1) + logpow(1 - Zanchored, beta - 1),
            0 <= Zanchored, Zanchored <= 1,
            alpha > 0,
            beta > 0)

        return logp


class DiscreteObsMJP(Continuous):
    def __init__(self, pi, Q, M, nObs, observed_jumps, T, *args, **kwargs):
        super(DiscreteObsMJP, self).__init__(dtype='int32', *args, **kwargs)
        self.pi = pi
        self.Q = Q
        self.M = M
        self.observed_jumps = observed_jumps
        self.T = T
        # compute all possible step sizes within observations, result like [1,2,3]
        step_sizes = np.unique(observed_jumps)
        step_sizes = step_sizes[step_sizes > 0]
        self.step_sizes = step_sizes
        self.mode = np.ones(nObs, dtype=np.int32)

        self.nObs = nObs
        # Long list of observation jumps, like [-1, 0,0,0,1,0,0,-1,0,2]
        self.obs_jump_ind = observed_jumps.copy()
        self.obs_jump_ind[observed_jumps == 0] = -1
        for ind in range(len(step_sizes)):
            self.obs_jump_ind[observed_jumps == step_sizes[ind]] = ind


    def computeC(self, S):
        M = self.M
        n_step_sizes = len(self.step_sizes)

        obs_jump_ind = TT.as_tensor_variable(self.obs_jump_ind, 'obs_jump_ind')
        # Here is the exp in jumps
        tau_ind = obs_jump_ind[1:] * M * M
        keep_jumps = (tau_ind >= 0).nonzero()

        jump_from_ind = S[:-1] * M
        jump_to_ind = S[1:]

        flat_ind = (tau_ind + jump_from_ind + jump_to_ind)[keep_jumps]
        flat_ind_counts = bincount(flat_ind, minlength=n_step_sizes * M * M)

        C = flat_ind_counts.reshape(shape=np.array([n_step_sizes, M, M]))

        return C

    # @profilingUtil.timefunc
    def logp(self, S):
        l = 0.0

        # add prior
        pi = self.pi
        # Get time 0 states
        zeroIndices = np.roll(self.T.cumsum(), 1)
        zeroIndices[0] = 0
        zeroIndices = zeroIndices.astype('int32')
        l += TT.sum(TT.log(pi[S[zeroIndices]]))
        # l += TT.sum(TT.log(pi[S[:,0]]))

        # add likelihood
        Q = self.Q
        step_sizes = self.step_sizes

        # import pdb; pdb.set_trace()
        C = self.computeC(S)

        n_step_sizes = len(self.step_sizes)
        for i in range(0, n_step_sizes):
            tau = step_sizes[i]
            P = TT.slinalg.expm(tau * Q)

            stabilizer = TT.tril(TT.alloc(0.0, *P.shape) + 0.1, k=-1)
            logP = TT.log(P + stabilizer)

            # compute likelihood in terms of P(tau)
            l += TT.sum(C[i, :, :] * logP)
        return l


from theano.compile.ops import as_op


@as_op(itypes=[TT.dscalar, TT.lscalar, TT.dmatrix, TT.dmatrix, TT.bmatrix, TT.ivector, TT.lvector], otypes=[TT.dscalar])
def logp_numpy_comorbidities(l, nObs, B0, B, X, S, T):
    logLike = np.array(0.0)

    # Unwrap t=0 points for B0
    zeroIndices = np.roll(T.cumsum(), 1)
    zeroIndices[0] = 0;
    zeroIndices = zeroIndices.astype('int32')

    # import pdb; pdb.set_trace()

    # Likelihood from B0 for X=1 and X=0 cases
    logLike += (X[zeroIndices] * np.log(B0[:, S[zeroIndices]]).T).sum()
    # logLike += (X[zeroIndices]*np.log(B0[:,S[zeroIndices]]).T).sum()
    logLike += ((1 - X[zeroIndices]) * np.log(1. - B0[:, S[zeroIndices]]).T).sum()

    stateChange = S[1:] - S[:-1]
    # Don't consider t=0 points
    stateChange[zeroIndices[1:] - 1] = 0
    changed = np.nonzero(stateChange)[0] + 1


    # A change can only happen from 0 to 1 given our assumptions
    logLike += ((X[changed] - X[changed - 1]) * np.log(B[:, S[changed]]).T).sum()
    logLike += (((1 - X[changed]) * (1 - X[changed - 1])) * np.log(1. - B[:, S[changed]]).T).sum()

    return logLike




class Comorbidities(Continuous):
    def __init__(self, S, B0, B, T, shape, *args, **kwargs):
        super(Comorbidities, self).__init__(shape=shape, dtype='int8', *args, **kwargs)
        X = np.ones(shape, dtype='int8')
        self.nObs = shape[0]
        self.K = shape[1]
        self.T = T
        self.S = S
        self.B0 = B0
        self.B = B
        self.mode = X

    def logp(self, X):
        l = np.float64(0.0)
        # l = logp_theano_comorbidities(l, self.nObs, self.B0, self.B, X, self.S, self.T)

        # Unwrap t=0 points for B0
        zeroIndices = np.roll(self.T.cumsum(), 1)
        zeroIndices[0] = 0
        zeroIndices = zeroIndices.astype('int32')


        # Likelihood from B0 for X=1 and X=0 cases
        l += (X[zeroIndices] * TT.log(self.B0[:, self.S[zeroIndices]]).T).sum()
        l += ((1 - X[zeroIndices]) * TT.log(1. - self.B0[:, self.S[zeroIndices]]).T).sum()

        stateChange = self.S[1:] - self.S[:-1]
        # Don't consider t=0 points

        stateChange = TT.set_subtensor(stateChange[zeroIndices[1:] - 1], 0)
        changed = TT.nonzero(stateChange)[0] + 1

        # A change can only happen from 0 to 1 given our assumptions
        l += ((X[changed] - X[changed - 1]) * TT.log(self.B[:, self.S[changed]]).T).sum()
        l += (((1 - X[changed]) * (1 - X[changed - 1])) * TT.log(1. - self.B[:, self.S[changed]]).T).sum()


        return l

def new_logp_theano_claims(l, nObs, T, Z, L, X, O, posMask):
    # import pdb; pdb.set_trace()

    Z_on = Z.T[O.T]
    denomLikelihood = (1. - L[O.T]) * (1. - X[np.newaxis, :, :] * (Z_on)).prod(axis=2)
    numLikelihood = (1. - denomLikelihood.T) * posMask + (1. - posMask)
    denomLikelihood = denomLikelihood.T * posMask + (1. - posMask)
    totalTerm = TT.log(1. - L).sum() * nObs + TT.log(1. - X[:, np.newaxis, :] * Z.T[np.newaxis, :, :]).sum()

    logLike = TT.log(numLikelihood).sum() - TT.log(denomLikelihood).sum() + totalTerm

    return logLike


class Claims(Continuous):
    def __init__(self, X, Z, L, T, D, O_input, shape, *args, **kwargs):
        super(Claims, self).__init__(shape=shape, dtype='int32', *args, **kwargs)
        self.X = X
        self.nObs = shape[0]
        # self.N = shape[2]
        self.Z = Z
        self.L = L
        self.T = T

        # Hacky way to do this by adding a -1 column that we then throw out
        self.pos_O_idx = np.zeros((self.nObs, D + 1), dtype='int8')
        self.pos_O_idx[np.arange(self.nObs), O_input.T] = 1
        self.pos_O_idx = self.pos_O_idx[:, :-1]

        self.O = O_input.astype('int16')
        self.posMask = (O_input != -1).astype('int8')

        O = np.ones(shape, dtype='int32')
        self.mode = O

    def newlogp(self, O):
        logLike = np.array(0.0)
        logLike = new_logp_theano_claims(TT.as_tensor_variable(logLike), TT.as_tensor_variable(self.nObs),
                                         TT.as_tensor_variable(self.T), self.Z, self.L, self.X,
                                         TT.as_tensor_variable(self.O), TT.as_tensor_variable(self.posMask))
        return logLike

    def logp(self, O):
        logLike = np.array(0.0)
        # import pdb; pdb.set_trace()
        logLike = logp_theano_claims(TT.as_tensor_variable(logLike), self.nObs,
                                     TT.as_tensor_variable(self.T), self.Z, self.L, self.X,
                                     TT.as_tensor_variable(self.pos_O_idx))
        return logLike


def logp_theano_claims(l, nObs, T, Z, L, X, O_on):
    # O_on = O_on.astype(np.bool)
    # tempVec is 1-X*Z
    tempVec = (1. - X.reshape((nObs, 1, X.shape[1])) * (Z.T).reshape((1, Z.shape[1], Z.shape[0])))
    # Add the contribution from O = 1
    logLike = TT.log(
        1 - (1 - TT.tile(L[np.newaxis, :], (nObs, 1))[O_on.nonzero()]) * TT.prod(tempVec[O_on.nonzero()], axis=1,
                                                                                 no_zeros_in_input=True)).sum()
    # Add the contribution from O = 0
    logLike += TT.log(
        (1 - TT.tile(L[np.newaxis, :], (nObs, 1))[(1 - O_on).nonzero()]) * TT.prod(tempVec[(1 - O_on).nonzero()],
                                                                                   axis=1,
                                                                                   no_zeros_in_input=True)).sum()
    # logLike += TT.log((1-TT.tile(L[np.newaxis,:],(nObs,1))[(1-O_on).nonzero()])*tempVec[(1-O_on).nonzero()].prod(axis=1)).sum()

    return logLike


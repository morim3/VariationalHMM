from abc import abstractmethod

import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax import lax, jit
from jax.scipy.special import digamma


class HMMBase:
    def __init__(self, ):
        pass

    def fit(self, obs):

        obs_log_probs = self.obs_log_prob(obs)
        trans_log_prob = self.trans_log_prob()
        initial_log_prob = self.initial_log_prob()

        forward, backward, log_scaling_factors = self._e_step(obs_log_probs, initial_log_prob, trans_log_prob)

        self._maximize_transitions(forward, backward, trans_log_prob, obs_log_probs, log_scaling_factors)

        self._maximize_obs_params(forward, backward)

        pass

    def predict(self, obs, viterbi=False):

        if not viterbi:
            return self.e_step(obs)

        else:
            return self.viterbi(obs)

    @abstractmethod
    def obs_log_prob(self, obs):
        raise NotImplementedError

    @abstractmethod
    def trans_log_prob(self):
        raise NotImplementedError

    @abstractmethod
    def initial_log_prob(self):
        raise NotImplementedError

    @staticmethod
    @jit
    def _forward_one_step(alpha, trans_log_prob, obs_log_prob, ):
        '''
        :param alpha: jnp array, shape (batch_size, hidden_num)
        :param trans_log_prob: jnp array, shape (hidden_num, hidden_num)
        :param obs_log_prob: jnp array, shape (batch_size, hidden_num)
        :return:
        '''
        log_prob = jnp.expand_dims(alpha, axis=2) + jnp.expand_dims(trans_log_prob, axis=0) \

        result = logsumexp(log_prob, axis=1) + obs_log_prob
        cond_log_likelihood = logsumexp(result, axis=1)
        return result, cond_log_likelihood

    @staticmethod
    @jit
    def _backward_one_step(beta, trans_log_prob, obs_log_prob, cond_log_likelihood):
        '''

        :param beta: jnp array, shape (batch_size, hidden_num)
        :param trans_log_prob: jnp array, shape (hidden, hidden)
        :param obs_log_prob: jnp array, shape (batch, hidden)
        :params cond_log_likelihood: jnp array, shape (batch)
        :return:
        '''
        log_prob = jnp.expand_dims(beta + obs_log_prob, axis=1) + jnp.expand_dims(trans_log_prob, axis=0)

        result = logsumexp(log_prob, axis=2) - jnp.expand_dims(cond_log_likelihood, axis=-1)

        return result

    @staticmethod
    @jit
    def forward_step(init_log_prob, obs_log_probs, trans_log_prob):

        def scan_fn(log_prob, obs_log_prob):
            result, partition = HMMBase._forward_one_step(log_prob, trans_log_prob,
                                                          obs_log_prob)
            normed_result = result - jnp.expand_dims(partition, axis=-1)
            return (
                normed_result,
                [normed_result, partition]
            )

        _, [forward_log_prob, cond_log_likelihood] = lax.scan(scan_fn, init_log_prob, obs_log_probs)
        return forward_log_prob, cond_log_likelihood

    @staticmethod
    @jit
    def backward_step(obs_log_probs, trans_log_prob, cond_log_likelihoods):

        @jit
        def scan_fn(beta, params):
            obs_log_prob = params[0]
            cond_log_likelihood = params[1]
            result = HMMBase._backward_one_step(beta, trans_log_prob,
                                                obs_log_prob, cond_log_likelihood)
            return (
                result,
                result
            )

        _, backward_log_prob = lax.scan(scan_fn, jnp.zeros_like(obs_log_probs[0]),
                                        [obs_log_probs, cond_log_likelihoods],
                                        reverse=True)
        return backward_log_prob

    @staticmethod
    @jit
    def _e_step(obs_log_probs, initial_log_prob, trans_log_prob):
        """
        :param obs_log_probs: shape(time, batch, hidden)
        """

        initial_forward = obs_log_probs[0] + jnp.expand_dims(initial_log_prob, axis=0)
        initial_log_likelihood = logsumexp(initial_forward, axis=-1)
        initial_forward = initial_forward - jnp.expand_dims(initial_log_likelihood, axis=1)

        forward_log_probs, cond_log_likelihoods = HMMBase.forward_step(initial_forward, obs_log_probs[1:],
                                                                       trans_log_prob)
        forward_log_probs = jnp.concatenate([initial_forward[jnp.newaxis], forward_log_probs])
        cond_log_likelihoods = jnp.concatenate([initial_log_likelihood[jnp.newaxis], cond_log_likelihoods])

        backward_log_probs = HMMBase.backward_step(obs_log_probs[1:], trans_log_prob, cond_log_likelihoods[1:])
        backward_log_probs = jnp.concatenate([backward_log_probs, jnp.zeros_like(obs_log_probs[0])[jnp.newaxis]])

        return forward_log_probs, backward_log_probs, cond_log_likelihoods

    def e_step(self, obs):
        obs_log_probs = self.obs_log_prob(obs)
        trans_log_prob = self.trans_log_prob()
        initial_log_prob = self.initial_log_prob()

        return self._e_step(obs_log_probs, trans_log_prob, initial_log_prob)

    @staticmethod
    @jit
    def _viterbi_one_step(a, b):
        """
        :param a: jnp array, shape (*, batch_size, hidden, hidden)
        :param b: jnp array, shape (*, batch_size, hidden, hidden)
        """
        if len(a) == 0:
            return a
        return jnp.max(jnp.expand_dims(a, axis=4) + jnp.expand_dims(b, axis=2), axis=3)

    @staticmethod
    @jit
    def _viterbi(initial_log_prob, trans_log_prob, obs_log_prob):
        """
        :param initial_log_prob: jnp array, shape (hidden)
        :param trnas_log_prob: jnp array, shape(hidden, hidden)
        :param obs_log_prob: jnp array, shape(time, batch_size, hidden)
        """
        obs_plus_trans = jnp.expand_dims(obs_log_prob[1:], axis=-1) + jnp.expand_dims(trans_log_prob, axis=(0, 1))
        obs_plus_initial = jnp.expand_dims(obs_log_prob[0], axis=-1) + jnp.expand_dims(initial_log_prob, axis=(0, 1, 2))
        elem = jnp.concatenate([obs_plus_initial, obs_plus_trans])
        forward = lax.associative_scan(HMMBase._viterbi_one_step, elem)

        obs_plus_trans = jnp.expand_dims(obs_log_prob[1:], axis=2) + jnp.expand_dims(trans_log_prob, axis=(0, 1,))
        elem = jnp.concatenate([obs_plus_trans, jnp.zeros_like(obs_plus_trans[0])[jnp.newaxis]], axis=0)
        backward = lax.associative_scan(HMMBase._viterbi_one_step, elem, reverse=True)

        result = jnp.argmax(forward[:, :, 0] + backward[:, :, :, 0], axis=2)
        return result

    def viterbi(self, obs):
        initial_log_prob = self.initial_log_prob()
        trans_log_prob = self.trans_log_prob()
        obs_log_prob = self.obs_log_prob(obs)

        return self._viterbi(initial_log_prob, trans_log_prob, obs_log_prob)


class VHMMBase(HMMBase):
    def __init__(self, init_state_prior, transition_prior):
        """

        :param init_state_prior: jnp array, dirichlet parameter of initial state prior, shape (hidden)
        :param transition_prior: jnp array, dirichlet parameter of state transition prior, shape (hidden, hidden)
        """
        super().__init__()
        self.init_state_prior = init_state_prior
        self.transition_prior = transition_prior

        self.init_state_posterior = init_state_prior
        self.transition_posterior = transition_prior

    def trans_log_prob(self):
        return digamma(self.transition_posterior) \
               - digamma(jnp.sum(self.transition_posterior, axis=1))[:, jnp.newaxis]

    def initial_log_prob(self):
        return digamma(self.init_state_prior) - digamma(jnp.sum(self.init_state_prior))

    def _maximize_transitions(self, forward, backward, transition_log_prob, obs_log_probs, log_scaling_factors):
        """

        :param forward: shape (time, batch, hidden)
        :param backward: (time, batch, hidden)
        :param transition_log_prob: (hidden, hidden)
        :param obs_log_probs: (time, batch, hidden)
        :param log_scaling_factors: (time, batch, )
        :return:
        """
        self.init_state_posterior = jnp.sum(jnp.exp(forward[0] + backward[0]), axis=0) + self.init_state_prior
        xi = forward[:-1][..., jnp.newaxis] + jnp.expand_dims(transition_log_prob, axis=(0, 1)) \
             + (obs_log_probs[1:] + backward[1:] + log_scaling_factors[1:][..., jnp.newaxis])[..., jnp.newaxis, :]
        self.transition_posterior = jnp.sum(jnp.exp(xi), axis=(0, 1)) + self.transition_prior

    @abstractmethod
    def obs_log_prob(self, obs):
        raise NotImplementedError

from abc import abstractmethod

import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax import lax, jit
from jax.lax import lgamma
from jax.scipy.special import digamma, xlogy


class HMMBase:
    def __init__(self, ):
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
        log_prob = jnp.expand_dims(alpha, axis=2) + \
            jnp.expand_dims(trans_log_prob, axis=0)

        result = logsumexp(log_prob, axis=1) + obs_log_prob
        return result

    @staticmethod
    @jit
    def _backward_one_step(beta, trans_log_prob, obs_log_prob, ):
        '''

        :param beta: jnp array, shape (batch_size, hidden_num)
        :param trans_log_prob: jnp array, shape (hidden, hidden)
        :param obs_log_prob: jnp array, shape (batch, hidden)
        :return:
        '''
        log_prob = jnp.expand_dims(
            beta + obs_log_prob, axis=1) + jnp.expand_dims(trans_log_prob, axis=0)
        result = logsumexp(log_prob, axis=2)
        return result

    @staticmethod
    @jit
    def forward_step(init_log_prob, obs_log_probs, trans_log_prob):

        def scan_fn(log_prob, obs_log_prob):
            result = HMMBase._forward_one_step(log_prob, trans_log_prob,
                                               obs_log_prob)
            return (
                result,
                result
            )

        _, forward_log_prob = lax.scan(scan_fn, init_log_prob, obs_log_probs)
        return forward_log_prob

    @staticmethod
    @jit
    def backward_step(obs_log_probs, trans_log_prob):

        @jit           
        def scan_fn(beta, obs_log_prob):
            result = HMMBase._backward_one_step(beta, trans_log_prob,
                                                obs_log_prob)
            return (
                result,
                result
            )

        _, backward_log_prob = lax.scan(scan_fn, jnp.zeros_like(obs_log_probs[0]),
                                        obs_log_probs,
                                        reverse=True)
        return backward_log_prob

    @staticmethod
    @jit
    def _e_step(obs_log_probs, initial_log_prob, trans_log_prob):
        """
        :param obs_log_probs: shape(time, batch, hidden)
        """

        initial_forward = obs_log_probs[0] + jnp.expand_dims(initial_log_prob, axis=0)

        forward_log_probs = HMMBase.forward_step(
            initial_forward, obs_log_probs[1:], trans_log_prob)
        forward_log_probs = jnp.concatenate(
            [initial_forward[jnp.newaxis], forward_log_probs])

        backward_log_probs = HMMBase.backward_step(
            obs_log_probs[1:], trans_log_prob)
        backward_log_probs = jnp.concatenate(
            [backward_log_probs, jnp.zeros_like(obs_log_probs[0])[jnp.newaxis]])

        return forward_log_probs, backward_log_probs

    @staticmethod
    @jit
    def _calc_gamma(forward, backward):
        log_gamma = forward + backward
        log_gamma = log_gamma - logsumexp(log_gamma, axis=-1, keepdims=True)
        return jnp.exp(log_gamma)

    @staticmethod
    @jit
    def _calc_xi(forward, backward, trans_log_prob, obs_log_probs):
        log_xi = forward[:-1][..., jnp.newaxis] + jnp.expand_dims(trans_log_prob, axis=(0, 1)) \
            + (obs_log_probs[1:] + backward[1:])[..., jnp.newaxis, :]
        log_xi = log_xi - logsumexp(log_xi, axis=(-1, -2), keepdims=True)
        return jnp.exp(log_xi)

    def e_step(self, obs):
        obs_log_probs = self.obs_log_prob(obs)
        trans_log_prob = self.trans_log_prob()
        initial_log_prob = self.initial_log_prob()

        forward, backward = HMMBase._e_step(
            obs_log_probs, initial_log_prob, trans_log_prob, )

        gamma = HMMBase._calc_gamma(forward, backward)
        xi = HMMBase._calc_xi(forward, backward, trans_log_prob, obs_log_probs)
        return gamma, xi

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
        obs_plus_trans = jnp.expand_dims(
            obs_log_prob[1:], axis=-1) + jnp.expand_dims(trans_log_prob, axis=(0, 1))
        obs_plus_initial = jnp.expand_dims(
            obs_log_prob[0], axis=-1) + jnp.expand_dims(initial_log_prob, axis=(0, 1, 2))
        elem = jnp.concatenate([obs_plus_initial, obs_plus_trans])
        forward = lax.associative_scan(HMMBase._viterbi_one_step, elem)

        obs_plus_trans = jnp.expand_dims(
            obs_log_prob[1:], axis=2) + jnp.expand_dims(trans_log_prob, axis=(0, 1,))
        elem = jnp.concatenate([obs_plus_trans, jnp.zeros_like(
            obs_plus_trans[0])[jnp.newaxis]], axis=0)
        backward = lax.associative_scan(
            HMMBase._viterbi_one_step, elem, reverse=True)

        result = jnp.argmax(forward[:, :, 0] + backward[:, :, 0], axis=2)
        return result

    def viterbi(self, obs):
        initial_log_prob = self.initial_log_prob()
        trans_log_prob = self.trans_log_prob()
        obs_log_prob = self.obs_log_prob(obs)

        return HMMBase._viterbi(initial_log_prob, trans_log_prob, obs_log_prob)


class MAPHMMBase(HMMBase):
    def __init__(self, init_state_prior, transition_prior):
        """

        :param init_state_prior: jnp array, dirichlet parameter of initial state prior, shape (hidden)
        :param transition_prior: jnp array, dirichlet parameter of state transition prior, shape (hidden, hidden)
        """
        super().__init__()

        self.init_state_posterior = init_state_prior
        self.transition_posterior = transition_prior

    def trans_log_prob(self):

        return jnp.log(self.transition_posterior)

    def initial_log_prob(self):
        return jnp.log(self.init_state_posterior)

    def maximize_transitions(self, gamma, xi):
        """
        :param gamma: shape (time, batch, hidden)
        :param xi: (time, batch, hidden, hidden)

        :return:
        """
        self.init_state_posterior = jnp.sum(gamma, axis=(0, 1)) 
        self.init_state_posterior = self.init_state_posterior / jnp.sum(self.init_state_posterior)
        self.transition_posterior = jnp.sum(xi, axis=(0, 1)) 
        self.transition_posterior = self.transition_posterior / jnp.sum(self.transition_posterior, axis=-1, keepdims=True)

    @abstractmethod
    def obs_log_prob(self, obs):
        raise NotImplementedError

 
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
        log_prob = digamma(self.transition_posterior)
        log_sum_prob = digamma(
            jnp.sum(self.transition_posterior, axis=-1))[..., jnp.newaxis]

        return log_prob - log_sum_prob

    def initial_log_prob(self):
        return digamma(self.init_state_posterior) - digamma(jnp.sum(self.init_state_posterior))

    def maximize_transitions(self, gamma, xi):
        """
        :param gamma: shape (time, batch, hidden)
        :param xi: (time, batch, hidden, hidden)

        :return:
        """
        self.init_state_posterior = jnp.sum(
            gamma, axis=(0, 1)) + self.init_state_prior
        self.transition_posterior = jnp.sum(
            xi, axis=(0, 1)) + self.transition_prior

    @abstractmethod
    def obs_log_prob(self, obs):
        raise NotImplementedError

    @staticmethod
    @jit
    def _kl_dirichlet_dirichlet(q, p):
        term1 = -lgamma(jnp.sum(p)) + jnp.sum(lgamma(p))
        term2 = lgamma(jnp.sum(q)) - jnp.sum(lgamma(q))
        term3 = jnp.sum((q - p) * (digamma(q) - digamma(jnp.sum(q))))
        return term1 + term2 + term3

    @staticmethod
    @jit
    def _kl_categorical(q, log_p):
        """

        :param q: (..., category_num)
        :param log_p: (..., category_num)
        :return:
        """
        return jnp.sum(xlogy(q, q) - q * log_p)

    def _kl_initial_state(self):
        return VHMMBase._kl_dirichlet_dirichlet(self.init_state_posterior, self.init_state_prior)

    def _kl_state_transition(self):
        return jnp.sum(jnp.array([VHMMBase._kl_dirichlet_dirichlet(q, p)
                                  for q, p in zip(self.transition_posterior, self.transition_prior)]))

    def _kl_hidden_state(self, gamma, xi):
        """

        :param gamma: shape (time, batch, hidden)
        :param xi: (time-1, batch, hidden, hidden)
        :return :
        """
        log_p = self.initial_log_prob()
        q = gamma[0]
        term1 = VHMMBase._kl_categorical(q, log_p)

        log_p = self.trans_log_prob()
        q = xi
        term2 = VHMMBase._kl_categorical(q, log_p)
        return term1 + term2

    def kl_partial(self, gamma, xi):
        return self._kl_hidden_state(gamma, xi) + self._kl_initial_state() + self._kl_state_transition() 

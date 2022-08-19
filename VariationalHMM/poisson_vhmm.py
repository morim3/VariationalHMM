from jax import jit
from jax.scipy.special import digamma, gammaln
import jax.numpy as jnp

from .vhmm_base import VHMMBase
import numpy as np


class PoissonVHMM(VHMMBase):

    def __init__(self, state_num, init_state_prior, transition_prior, poisson_prior,
                 iter_max=100, converge_threshold=1e-3):
        super().__init__(init_state_prior, transition_prior)
        self.state_num = state_num
        self.poisson_prior_a = jnp.array(poisson_prior[0])
        self.poisson_prior_b = jnp.array(poisson_prior[1])
        self.poisson_posterior_a = jnp.array(np.repeat(poisson_prior[0], state_num) + np.random.rand(state_num) * 1.)
        self.poisson_posterior_b = jnp.array(np.repeat(poisson_prior[1], state_num) + np.random.rand(state_num) * 1.)

        self.iter_max = iter_max
        self.converge_threshold = 1e-3

    @staticmethod
    @jit
    def _obs_log_prob(obs, poisson_posterior_a, poisson_posterior_b):
        term1 = obs[..., jnp.newaxis] \
                * (digamma(poisson_posterior_a) - jnp.log(poisson_posterior_b))[jnp.newaxis, jnp.newaxis]
        term2 = - (poisson_posterior_a / poisson_posterior_b)[jnp.newaxis, jnp.newaxis]
        term3 = - gammaln(obs)[..., jnp.newaxis]
        return term1 + term2 + term3

    def obs_log_prob(self, obs):
        return self._obs_log_prob(obs, self.poisson_posterior_a, self.poisson_posterior_b)

    def maximize_observations(self, obs, gamma):
        """

        :param obs: jnp array, shape (time, batch)
        :param gamma: jnp array, shape (time, batch, hidden)
        :return:
        """
        self.poisson_posterior_a = jnp.sum(obs[..., jnp.newaxis] * gamma, axis=(0, 1)) + self.poisson_prior_a
        self.poisson_posterior_b = jnp.sum(gamma, axis=(0, 1)) + self.poisson_prior_b

    def fit(self, obs):

        for i in range(self.iter_max):
            gamma, xi = self.e_step(obs)
            self.maximize_transitions(gamma, xi)
            self.maximize_observations(obs, gamma)
            elbo = self.elbo(obs)
            print(elbo)

        return gamma, self.viterbi(obs), elbo

    @staticmethod
    def _kl_lambda(q_a, q_b, p_a, p_b):
        """

        :param q_a: (hidden)
        :param q_b: (hidden)
        :param p_a: (1)
        :param p_b: (1)
        :return:
        """
        term1 = q_a * jnp.log(q_b) - gammaln(q_a) - q_a + (q_a - 1)*(digamma(q_a) - digamma(q_b))
        term2 = p_a * jnp.log(q_b) - gammaln(p_a) - p_b * q_a / q_b + (p_a - 1)*(digamma(q_a) - digamma(q_b))
        return jnp.sum(term2 - term1)

    def elbo(self, obs):
        gamma, xi = self.e_step(obs)

        return (jnp.sum(self.obs_log_prob(obs) * gamma)
                - self._kl_lambda(self.poisson_posterior_a,
                                  self.poisson_posterior_b,
                                  self.poisson_prior_a,
                                  self.poisson_prior_b)
                - self.kl_partial(gamma, xi))

from jax import jit
from jax.scipy.special import digamma, gammaln
import jax.numpy as jnp

from model.vhmm_base import VHMMBase


class PoissonVHMM(VHMMBase):

    def __init__(self, state_num, init_state_prior, transition_prior, poisson_prior,
                 iter_max=100, converge_threshold=1e-3):
        super().__init__(init_state_prior, transition_prior)
        self.state_num = state_num
        self.poisson_prior_a = poisson_prior[0]
        self.poisson_prior_b = poisson_prior[1]
        self.poisson_posterior_a = jnp.repeat(poisson_prior[0], state_num)
        self.poisson_posterior_b = jnp.repeat(poisson_prior[1], state_num)

        self.iter_max = iter_max
        self.converge_threshold = 1e-3

    @staticmethod
    @jit
    def _obs_log_prob(obs, poisson_posterior_a, poisson_posterior_b):
        term1 = obs * jnp.expand_dims(digamma(poisson_posterior_a) - jnp.log(poisson_posterior_b), axis=(0, 1))
        term2 = - (poisson_posterior_a / poisson_posterior_b)[jnp.newaxis, jnp.newaxis]
        term3 = - gammaln(obs)
        return term1 + term2 + term3

    def obs_log_prob(self, obs):
        return self._obs_log_prob(obs, self.poisson_posterior_a, self.poisson_posterior_b)

    def maximize_observations(self, obs, gamma):
        """

        :param obs: jnp array, shape (time, batch, hidden)
        :param gamma: jnp array, shape (time, batch, hidden)
        :return:
        """
        self.poisson_posterior_a = jnp.sum(obs * gamma, axis=(0, 1)) + self.poisson_prior_a
        self.poisson_posterior_b = jnp.sum(gamma, axis=(0, 1)) + self.poisson_prior_b

    def fit(self, obs):

        for i in range(self.iter_max):
            gamma, xi = self.e_step(obs)
            self.maximize_transitions(gamma, xi)
            self.maximize_observations(obs, gamma)
            elbo = self.elbo(obs)
            print(elbo)

        return gamma, self.viterbi(obs)


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
        print("dsfadf")
        return (jnp.sum(self.obs_log_prob(obs) * gamma)
                - self._kl_lambda(self.poisson_posterior_a,
                                  self.poisson_posterior_b,
                                  self.poisson_prior_a,
                                  self.poisson_prior_b)
                - self._kl_hidden_state(gamma, xi)
                - self._kl_initial_state()
                - self._kl_state_transition())

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

    def obs_log_prob(self, obs):
        term1 = obs * jnp.expand_dims(digamma(self.poisson_posterior_a) - jnp.log(self.poisson_posterior_b), axis=(0, 1))
        term2 = - (self.poisson_posterior_a / self.poisson_posterior_b)[jnp.newaxis, jnp.newaxis]
        term3 = - gammaln(obs)[..., jnp.newaxis]
        return term1 + term2 + term3

    def _maximize_observations(self, obs, gamma):
        """

        :param obs: jnp array, shape (time, batch, hidden)
        :param gamma: jnp array, shape (time, batch, hidden)
        :return:
        """
        self.poisson_posterior_a = jnp.sum(obs * gamma, axis=(0, 1)) + self.poisson_prior_a
        self.poisson_posterior_b = jnp.sum(gamma, axis=(0, 1)) + self.poisson_prior_b

    def fit(self, obs):
        pass

    def variational_lower_bound(self):
        pass


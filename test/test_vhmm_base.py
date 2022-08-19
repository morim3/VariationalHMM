from unittest import TestCase
import unittest
from VariationalHMM.vhmm_base import HMMBase, VHMMBase
import jax.numpy as jnp
import numpy as np


class TestVHMMBase(TestCase):

    def test_viterbi(self):
        test_obs = jnp.ones((10, 3, 2))
        result = HMMBase._viterbi(jnp.log(jnp.array([0.2, 0.8])), jnp.log(jnp.array([[1, 0], [0, 1]])), test_obs)

        self.assertEqual(result.shape[0], 10)
        self.assertEqual(result.shape[1], 3)
        self.assertEqual(result[0, 0], 1)

        test_obs = jnp.ones((10, 3, 2))
        result = HMMBase._viterbi(jnp.log(jnp.array([0.6, 0.4])), jnp.log(jnp.array([[1, 0], [0, 1]])), test_obs)
        self.assertEqual(result[0, 0], 0)

    def test_expectation(self):
        test_obs = jnp.ones((100, 3, 2))
        forward, backward = HMMBase._e_step(test_obs, jnp.log(jnp.array([0.8, 0.1])),
                                               jnp.log(jnp.array([[0.1, 0.4], [0., 1.0]])), )

        hidden = HMMBase._calc_gamma(forward, backward)
        self.assertAlmostEqual(hidden[-1, 0, 1], 1, delta=1e-5)

        test_obs = np.zeros((10, 3, 2))
        test_obs[:5, :, 0] = -1.
        test_obs[5:, :, 0] = 1.
        test_obs = jnp.array(test_obs)
        forward, backward = HMMBase._e_step(test_obs, jnp.log(jnp.array([0.5, 0.5])),
                                               jnp.log(jnp.array([[0.9, 0.1], [0.1, 0.9]])), )


        hidden = HMMBase._calc_gamma(forward, backward)
        self.assertTrue(hidden[0, 0, 1] > hidden[0, 0, 0])
        self.assertTrue(hidden[-1, 0, 1] < hidden[-1, 0, 0])
        self.assertTrue(jnp.all(jnp.abs(jnp.sum(hidden, axis=-1) - jnp.ones_like(hidden[..., 0])) < 1e-6))


if __name__ == "__main__":
    unittest.main()

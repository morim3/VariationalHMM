from unittest import TestCase
import unittest
from model.vhmm_base import HMMBase
import jax.numpy as jnp


class TestVHMMBase(TestCase):

    def test_viterbi(self):
        test_obs = jnp.ones((1000, 100, 2))
        result = HMMBase._viterbi(jnp.log(jnp.array([0.2, 0.8])), jnp.log(jnp.array([[1, 0], [0, 1]])), test_obs)  
        
        self.assertEqual(result.shape[0], 10)
        self.assertEqual(result.shape[1], 3)
        self.assertEqual(result[0, 0], 1)

        test_obs = jnp.ones((10, 3, 2))
        result = HMMBase._viterbi(jnp.log(jnp.array([0.6, 0.4])), jnp.log(jnp.array([[1, 0], [0, 1]])), test_obs) 
        self.assertEqual(result[0, 0], 0)

    def test_expectation(self):
        test_obs = jnp.ones((10, 3, 2))
        result = HMMBase._e_step(test_obs, jnp.log(jnp.array([[1, 0], [0, 1]])), jnp.log(jnp.array([0.1, 0.9])))
        print(result)




if __name__ == "__main__":
    unittest.main()

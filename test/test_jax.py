import unittest
from unittest import TestCase
from jax import lax
import jax.numpy as jnp


class TestJaxFunction(TestCase):
    def test_scan(self):
        def fn_scan(sum, add):
            return sum + add

        result = lax.associative_scan(fn_scan, jnp.array([1, 2, 3, 4]))

        self.assertEqual(result[2], 6)

    def test_scan_multidim(self):
        def fn_scan(a, b):
            print("a", a)
            print("b", b)
            return a+b

        result = lax.associative_scan(fn_scan, jnp.ones((5, 3)))
        self.assertEqual(result[-1][-1], 5)

    def test_scan_multi(self):
        def fn_scan(sum, add):
            return (
                sum + add,
                [sum + add, sum*sum]
            )

        result, cum = lax.scan(fn_scan, jnp.array([0]), jnp.array([1, 2, 3, 4]))


if __name__ == "__main__":
    unittest.main()

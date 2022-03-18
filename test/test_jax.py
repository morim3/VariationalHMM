from unittest import TestCase
from jax import lax


class TestJaxFunction(TestCase):
    def test_scan(self):
        def fn_scan(sum, add):
            return (
                sum + add,
                sum + add
            )

        result, cum = lax.scan(fn_scan, 0, [1, 2, 3, 4])

        print(result, cum)


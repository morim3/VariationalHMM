from glob import glob
from os.path import basename
from os.path import splitext

from setuptools import setup
from setuptools import find_packages


def _requires_from_file(filename):
    return open(filename).read().splitlines()

setup(
    name="VariationalHMM",
    version="0.1.0",
    description="Jax-Implementation of Variational Hidden Markov Model",
    author="Kohei Morimoto",
    url="https://github.com/morim3/VariationalHMM",
    packages=["VariationalHMM"],
    include_package_data=True,
    zip_safe=False,
    install_requires=_requires_from_file('requirements.txt'),
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "pytest-cov"]
)

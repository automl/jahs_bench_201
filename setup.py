import os
from setuptools import setup, find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "JAHS-Bench-MF",
    version = "0.0.2",
    author = "Archit Bansal",
    author_email = "bansala@cs.uni-freiburg.de",
    description = ("An API to access the JAHS-Bench benchmark for "
        "Multi-Fidelity, Multi-Objective joint NAS and HPO."),
    license = "MIT",
    keywords = "NAS HPO Benchmark Multi-Fidelity Multi-Objective",
    url = "https://github.com/automl/jahs_bench_mf",
    packages=find_packages('JAHS-Bench-MF', include=['jahs_bench', 'jahs_bench.lib']),
    long_description=read('README.md'),
    package_dir={
        '': 'JAHS-Bench-MF',
    },
    python_requires=">=3.7",
    install_requires=[
        "numpy >=1.21",
        "pandas >=1.4",
        "scipy >=1.7",
        "scikit-learn >=1.0",
        "xgboost >= 1.5",
        "ConfigSpace >= 0.4"
    ]
)

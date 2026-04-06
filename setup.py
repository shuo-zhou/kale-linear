#!/usr/bin/env python3
# import re
from io import open
from os import path

from setuptools import find_packages, setup

# Dependencies with options for different user needs. If updating this, you may need to update docs/requirements.txt too.
# If option names are changed, you need to update the installation guide at docs/source/installation.md respectively.
# Not all have a min-version specified, which is not uncommon. Specify when known or necessary (e.g. errors).


# Core dependencies frequently used in the kalelinear API
install_requires = [
    "cvxopt",
    "numpy",
    "osqp",
    "pandas",
    "scikit-learn",
    "scipy",
    "tensorly",
]

# Optional dependency for tensor input/output interoperability.
torch_requires = [
    "torch",
]

# Dependencies for all examples and tutorials
example_requires = [
    "ipykernel",
    "ipython",
    "matplotlib",
    "nilearn",
    "seaborn",
    "yacs>=0.1.7",
]

# Full dependencies except for development
full_requires = install_requires + example_requires

# Additional dependencies for development
dev_requires = full_requires + [
    "black==19.10b0",
    "coverage",
    "flake8",
    "flake8-print",
    "ipywidgets",
    "isort",
    "m2r",
    "mypy",
    "nbmake>=0.8",
    "nbsphinx",
    "nbsphinx-link",
    "nbval",
    "pre-commit",
    "pytest",
    "pytest-cov",
    "recommonmark",
    "sphinx",
    "sphinx-rtd-theme",
]


# Get version
def read(*names, **kwargs):
    with open(
        path.join(path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8"),
    ) as fp:
        return fp.read()


setup(
    name="kalelinear",
    version="0.1.0a1",
    description="Non-deep knowledge-aware machine learning from multiple sources/views in Python",
    url="https://github.com/pykale/kale-linear",
    author="Shuo Zhou",
    author_email="shuo.zhou@sheffield.ac.uk",
    license="MIT License",
    packages=find_packages(exclude=("tests*", "examples*", "docs*")),
    install_requires=install_requires,
    extras_require={
        "torch": torch_requires,
        "full": full_requires + torch_requires,
        "dev": dev_requires + torch_requires,
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)

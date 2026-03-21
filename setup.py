#!/usr/bin/env python3
# import re
from io import open
from os import path

from setuptools import setup

# Dependencies with options for different user needs. If updating this, you may need to update docs/requirements.txt too.
# If option names are changed, you need to update the installation guide at docs/source/installation.md respectively.
# Not all have a min-version specified, which is not uncommon. Specify when known or necessary (e.g. errors).
# The recommended practice is to install PyTorch from the official website to match the hardware first.
# To work on graphs, install torch-geometric following the official instructions at https://github.com/pyg-team/pytorch_geometric#installation

# Key reference followed: https://github.com/pyg-team/pytorch_geometric/blob/master/setup.py

# Core dependencies frequently used in the Kalinear API
install_requires = [
    "cvxopt",  # sure
    "numpy>=1.18.0",  # sure
    "osqp",  # sure
    "pandas",  # sure
    "scikit-learn>=0.23.2",  # sure
    "scipy>=1.5.4",  # in factorization API only
]

# Dependencies for all examples and tutorials
example_requires = [
    "ipykernel",
    "ipython",
    "matplotlib<=3.5.2",
    "nilearn",
    "Pillow",
    "PyTDC",
    "seaborn",
    "torchsummary>=1.5.0",
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
    name="kalinear",
    version="0.1.0a1",
    description="A Transfer Learning Python package",
    url="https://github.com/sz144/TPy",
    author="Shuo Zhou",
    author_email="szhou20@sheffield.ac.uk",
    license="MIT License",
    packages=["kalinear"],
    install_requires=["numpy", "scipy", "pandas", "scikit-learn", "cvxopt", "osqp"],
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

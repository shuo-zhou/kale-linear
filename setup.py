#!/usr/bin/env python3
import re
from io import open
from os import path

from setuptools import find_packages, setup

# Dependencies with options for different user needs. If updating this, you may need to update docs/requirements.txt too.
# If option names are changed, you need to update the installation guide at docs/source/installation.md respectively.
# Not all have a min-version specified, which is not uncommon. Specify when known or necessary (e.g. errors).
# The recommended practice is to install PyTorch from the official website to match the hardware first.
# To work on graphs, install torch-geometric following the official instructions at https://github.com/pyg-team/pytorch_geometric#installation

# Key reference followed: https://github.com/pyg-team/pytorch_geometric/blob/master/setup.py

# Core dependencies frequently used in PyKale Core API
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
    with open(path.join(path.dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8"),) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


readme = open("README.md").read()
version = find_version("kale", "__init__.py")


# Run the setup
setup(
    name="pydale",
    version="0.1.0a1",
    description="Domain-Aware Learning in Python",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/sz144/pydale",
    author="Shuo Zhou",
    author_email="shuo.zhou@sheffield.ac.uk",
    license="MIT License",
    packages=["pydale"],
    packages=find_packages(exclude=("docs", "examples", "tests")),
    python_requires=">=3.7,<3.10",
    install_requires=install_requires,  # ["numpy", "scipy", "pandas", "scikit-learn", "cvxopt", "osqp", "pytest", "pykale",],
    python_requires=">=3.8,<3.10",
    setup_requires=["setuptools==59.5.0"],
    keywords="machine learning, pytorch, deep learning, multimodal learning, transfer learning",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Software Development :: Libraries",
        "Natural Language :: English",
    ],
)

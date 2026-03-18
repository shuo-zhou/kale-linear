from setuptools import setup

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
    python_requires=">=3.6",
)

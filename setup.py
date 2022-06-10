from setuptools import setup, find_packages

requirements = [
    "numpy==1.22.3",
    "dysts==0.1",
    "matplotlib==3.5.0",
    "torch==1.10.0",
    "gpytorch==1.5.1",
    "scikit_learn==1.0.1",
    "pandas==1.3.4",
    "gif==3.0.0",
    "wbml==0.3.14",
    "wandb==0.12.14",
    "kmeans-pytorch==0.3",
    "tqdm==4.64.0",
    "openpyxl==3.0.10",
]

setup(
    name="npdgp",
    author="Magnus Ross & Thomas M. McDonald",
    packages=["npdgp"],
    description="Implementation of NP-CGP and NP-DGP.",
    long_description=open("README.md").read(),
    install_requires=requirements,
    python_requires=">=3.9",
)

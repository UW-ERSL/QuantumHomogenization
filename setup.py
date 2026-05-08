from setuptools import setup, find_packages

setup(
    name="qhomogenize",
    version="0.1.0",
    description="Quantum block-encoding for computational homogenization",
    author="Krishnan Suresh",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=2.0",
        "scipy>=1.10",
        "qiskit>=2.4",
        "qiskit-aer",
        "matplotlib",
    ],
    extras_require={
        "dev": ["pytest>=7.0"],
    },
)

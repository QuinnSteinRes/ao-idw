from setuptools import setup, find_packages

setup(
    name="intensityPINNIDW",
    version="0.1.0",
    description="Inverse-Dirichlet Weighted Physics-Informed Neural Networks",
    author="Quinn",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "tensorflow>=2.16.0",
        "numpy<2.0",
        "scipy",
        "matplotlib",
        "pyDOE",
        "seaborn",
        "pyyaml",
    ],
)

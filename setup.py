from setuptools import setup, find_packages

setup(
    name="lootcapital",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.2.0",
        "numpy>=2.2.0",
        "scikit-learn>=1.6.0",
        "xgboost>=3.0.0",
        "matplotlib>=3.10.0",
        "seaborn>=0.13.0",
        "joblib>=1.5.0"
    ],
    python_requires=">=3.8",
)

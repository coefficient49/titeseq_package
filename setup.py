from setuptools import setup, find_packages

setup(
    name="titeseq",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "click",
        "pandas",
        "numpy",
        "scipy",
        "torch",
    ],
    entry_points={
        "console_scripts": [
            "titeseq=titeseq.cli:main",
        ],
    },
)
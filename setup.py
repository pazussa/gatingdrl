from setuptools import setup, find_packages
from pathlib import Path

setup(
    name="gatingdrl",
    version="0.1.0",
    description="Time Sensitive Networks Scheduling with Deep Reinforcement Learning",
    long_description=(Path(__file__).with_name("README.md").read_text(encoding="utf-8")),
    long_description_content_type="text/markdown",
    url="https://github.com/juanpussa/gatingdrl",
    author="Juan Paz",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "networkx>=3.0",
        "matplotlib",
        "pandas",
        "numpy",
        "gymnasium>=0.28.1",
        "sb3-contrib>=2.0.0",
        "stable-baselines3>=2.0.0",
        "plotly>=5.14.0",
        "seaborn>=0.12.0",            #  usada en tools/analyze_training.py
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
)



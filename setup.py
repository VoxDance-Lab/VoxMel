#!/usr/bin/env python3

import os

from setuptools import find_packages
from setuptools import setup

with open(os.path.join(os.path.dirname(__file__), "requirements.txt")) as f:
    install_requires = [line.strip() for line in f if line.strip() and not line.startswith("#")]


dirname = os.path.dirname(__file__)
version_file = os.path.join(dirname, "voxcore", "version.txt")
with open(version_file, "r") as f:
    version = f.read().strip()

setup(
    name="voxcore",
    version=version,
    description="VoxMel: Qwen2.5-LLM conditioned multi-scale mel diffusion TTS (train & inference)",
    long_description=open(os.path.join(dirname, "README.md"), encoding="utf-8").read()
    if os.path.isfile(os.path.join(dirname, "README.md"))
    else "",
    long_description_content_type="text/markdown",
    license="The MIT License",
    packages=find_packages(include=["voxcore*"]),
    package_data={"voxcore": ["version.txt"]},
    install_requires=install_requires,
    extras_require={"web": ["gradio==5.50.0"]},
    python_requires=">=3.10,<3.13",
    entry_points={"console_scripts": ["voxmel=voxcore.cli:main"]},
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
)

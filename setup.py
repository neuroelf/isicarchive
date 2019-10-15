#!/usr/bin/env python

import setuptools

with open('isicarchive/version.py', 'r') as fh:
    __version__ = fh.read().strip().split('=')[-1].strip(' \'"')

with open("README.md", "r") as fh:
    long_description = fh.read()

requires = [
    'imageio>=2.5.0',
    'ipython>=7.1.0',
    'ipywidgets>=7.1.0',
    'matplotlib>=3.1.0',
    'numba>=0.45.1',
    'numpy>=1.16.2',
    'scipy>=1.3.1',
    'requests>=2.22.0',
]

setuptools.setup(
    name="isicarchive",
    version=__version__,
    description="ISIC Archive API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jochen Weber",
    author_email="weberj3@mskcc.org",
    url="https://github.com/neuroelf/isicarchive",
    packages=setuptools.find_packages(),
    package_dir={'isicarchive': 'isicarchive'},
    python_requires=">=3.6",
    install_requires=requires,
    include_package_data=True,
    license='MIT',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
)

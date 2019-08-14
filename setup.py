#!/usr/bin/env python

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requires = [
    'imageio>=2.5.0',
    'numpy>=1.16.2',
    'requests>=2.22.0',
]

setuptools.setup(
    name="isicarchive",
    version="0.2.0",
    description="ISIC Archive API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jochen Weber",
    author_email="weberj3@mskcc.org",
    url="https://github.com/neuroelf/isic-archive",
    packages=setuptools.find_packages(),
    package_dir={'isicarchive': 'isicarchive'},
    python_requires=">=3.6",
    install_requires=requires,
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

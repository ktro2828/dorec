#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import unicode_literals
import os.path as osp
from setuptools import setup, find_packages

try:
    with open("README.md") as f:
        readme = f.read()
except IOError:
    readme = ""


def _requires_from_file(filename):
    return open(filename).read().splitlines()


# Version
here = osp.dirname(osp.abspath(__file__))
version = next((line.split("=")[1].strip().replace("'", "")
                for line in open(osp.join(here, "dorec", "__init__.py"))
                if line.startswith("__version__ = ")), "0.0.1")

setup(
    name="dorec",
    version=version,
    url="https://github.com/ktro2828/dorec",
    author="ktro2828",
    author_email="ktro310115@gmail.com",
    maintainer="ktro2828",
    maintainer_email="ktro310115@gmail.com",
    description="",
    long_description=readme,
    packages=find_packages(),
    install_requires=_requires_from_file("requirements/requirements.txt"),
    license="MIT",
    classifiers=[
        "Programing Language :: Python :: 2.7",
        "Programing Language :: Python :: 3.6",
        "Programing Language :: Python :: 3.7",
        "Programing Language :: Python :: 3.8",
        "Programing Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License"
    ],
    entry_points="""
        # -*- Entry points: -*-
        [console_scripts]
        pkgdep = pypipkg.scripts.command:main
    """
)

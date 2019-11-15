#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
todo
"""

import setuptools
from pip.req import parse_requirements
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hanabi",
    version="1.0",
    author="Sascha",
    author_email="sascha.lange@campus.tu-berlin.de",
    description="hanabi-training-suite",
#    long_description=long_description,
#    long_description_content_type="text/markdown",
    url="https://github.com/hellovertex/hanabi",
    install_requires=parse_requirements("requirements.txt", session="hack"),
#    packages=setuptools.find_packages(),
    include_package_data=True
#    classifiers=(
#        "Programming Language :: Python :: 3",
#        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
#        "Operating System :: OS Independent",
#    ),
#    entry_points="""[console_scripts]
#            dlc=dlc:main""",
)

#https://www.python.org/dev/peps/pep-0440/#compatible-release

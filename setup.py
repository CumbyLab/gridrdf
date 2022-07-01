#!/usr/bin/env python

__author__ = "James Cumby"
__copyright__ = "Copyright James Cumby (2022)"
__version__ = "0.1.0"
__maintainer__ = "James Cumby"
__email__ = "james.cumby@ed.ac.uk"
__date__ = "June 30 2022"

from setuptools import setup
import os
import unittest

module_dir = os.path.dirname(os.path.abspath(__file__))

# Requirements
reqs_file = os.path.join(module_dir, "requirements.txt")
with open(reqs_file) as f:
    reqs_raw = f.read()
reqs_list = [r.replace("==", ">=") for r in reqs_raw.split("\n")]

# Optional requirements
# Approach borrowed from matminer, https://github.com/hackingmaterials/matminer/blob/main/setup.py
extras_file = os.path.join(module_dir, "requirements-optional.txt")
with open(extras_file) as f:
    extras_raw = f.read()
extras_raw = [r for r in extras_raw.split("##") if r.strip() and "#" not in r]
extras_dict = {}
for req in extras_raw:
    items = [i.replace("==", ">=") for i in req.split("\n") if i.strip()]
    dependency_name = items[0].strip()
    dependency_reqs = [i.strip() for i in items[1:] if i.strip()]
    extras_dict[dependency_name] = dependency_reqs
extras_list = [r for d in extras_dict.values() for r in d]

if __name__ == "__main__":
    setup(
        name="gridrdf",
        version="0.1.0",
        description="Grouped representation of interatomic distances",
        long_description=open(os.path.join(module_dir, "README.md")).read(),
        long_description_content_type="text/markdown",
        url="https://git.ecdf.ed.ac.uk/funcmatgroup/gridrdf",
        author="James Cumby",
        author_email="james.cumby@ed.ac.uk",
        maintainer="James Cumby",
        maintainer_email="james.cumby@ed.ac.uk",
        license="MIT",
        packages=["gridrdf"],
        package_data={
        },
        zip_safe=False,
        test_suite="tests",
        install_requires=reqs_list,
        extras_require=extras_dict,
        classifiers=[
            "Programming Language :: Python",
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "Operating System :: OS Independent",
            "Topic :: Scientific/Engineering",
            "License :: OSI Approved :: MIT License",
        ],
        python_requires='>=3.6',
    )


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
        install_requires=[
            "scipy",
            "numpy",
            "pymatgen",
            "pandas",
            "sklearn",
            "pyemd",
            "matminer",
            
        ],
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


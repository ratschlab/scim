#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

setup_requirements = ['pytest-runner']

with open('requirements.txt') as f:
    requirements = list(f.readlines())

test_requirements = ['pytest']

setup(
    author="starks",
    author_email='stefan.stark@inf.ethz.ch',
    description="SCIM: Single-Cell Integration via Matching",
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords='scim',
    name='scim',
    packages=find_packages(include=['scim']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/ratschlab/scim',
    version='0.1.0',
    zip_safe=False,
)

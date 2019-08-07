#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = [
        'anndata', 'hvplot', 'pandas', 'numpy', 'holoviews', 'scipy'
]

extras_require = {
}

setup_requirements = [

]

test_requirements = [

]

setuptools.setup(
    name='scplot',
    version='0.0.6',
    author="Joshua Gould",
    author_email='jgould@broadinstitute.org',
    description="Single cell plotting",
    url='https://github.com/klarman-cell-observatory/scPlot',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(include=['scplot']),
    include_package_data=True,
    install_requires=requirements,
    license="BSD license",
    zip_safe=False,
    keywords='wot',
    classifiers=[
            'License :: OSI Approved :: BSD License',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'Natural Language :: English',
            'Operating System :: MacOS :: MacOS X',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX :: Linux',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
            'Topic :: Scientific/Engineering :: Visualization',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements
)

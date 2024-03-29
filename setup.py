import os
# Setuptools should be preferred:
# * https://stackoverflow.com/a/25372045/5578392
# * https://stackoverflow.com/a/14753678/5578392
#
# https://stackoverflow.com/a/41573588/5578392
from setuptools import setup
from setuptools import find_packages

with open(os.path.join('./', 'VERSION')) as version_file:
    project_version = version_file.read().strip()

setup(
    name='paddle',
    version=project_version,
    description='Passau Data Science Deep Learning Environments',
    author='Chair of Data Science, University of Passau',
    author_email='julian.stier@uni-passau.de',
    url='https://gitlab.padim.fim.uni-passau.de/paddle/paddle',
    packages=find_packages(),
    license="GPL3 License",
    install_requires=['pip', 'setuptools>=18.0'],
    dependency_links=[],
    classifiers = [
        "Development Status :: 1 - Planning",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Cython",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
)
import os
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = 'ovnlp'
DESCRIPTION = 'Openvalue toolkit to use word vectors'
URL = 'https://github.com/Fanchouille/ovnlp'
EMAIL = 'francois.valadier@gmail.com'
AUTHOR = 'Francois Valadier'
VERSION = "0.0.2"

# What packages are required for this module to be executed?
REQUIRED = [
    "tqdm>=4.23.4",
    "gensim>=3.4.0",
    "pandas>=0.21.1",
    "nltk>=3.2.4",
    "numpy>=1.13.3",
    "requests>=2.18.4",
    "pathlib>=1.0.1"
]

# print(find_packages(exclude=('tests', 'fasttext/weights')))

here = os.path.abspath(os.path.dirname(__file__))

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    install_requires=REQUIRED,
    include_package_data=True,
    license='MIT',

)

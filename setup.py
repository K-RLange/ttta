from distutils.core import setup, Extension
from setuptools import find_packages
import numpy
from Cython.Distutils import build_ext
from pathlib import Path
import os

os.environ["C_INCLUDE_PATH"] = numpy.get_include()
VERSION = '0.0.1'
DESCRIPTION = 'Tools for temporal text analysis: A Python package providing diachronic tools for text analysis.'

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="ttta",
    version=VERSION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Kai-Robin Lange, Lars Grönberg, Niklas Benner, Imene Kolli, Aymane Hachcham, Jonas Rieger and Carsten Jentsch",
    author_email="<kalange@statistik.tu-dortmund.de>",
    description=DESCRIPTION,
    packages=find_packages(),
    url="https://github.com/K-RLange/ttta",
    install_requires=['nltk', 'Cython', "pandas", "gensim", "matplotlib", "wordcloud", "spacy", "joblib",
                      "scipy", "numpy", "tqdm", "seaborn"],
    keywords=['python', 'nlp', 'diachronic embeddings', 'dynamic topic modelling', 'document scaling'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: Implementation :: CPython",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("src.ttta.methods.LDA.lda_gibbs",
                 sources=["src/ttta/methods/LDA/lda_gibbs.pyx", "src/ttta/methods/LDA/lda_gibbs.c"],
                 include_dirs=[numpy.get_include()])],
)

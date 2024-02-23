from distutils.core import setup, Extension
import numpy
from Cython.Distutils import build_ext
import os

os.environ["C_INCLUDE_PATH"] = numpy.get_include()
setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("ttta.methods.LDA.lda_gibbs",
                 sources=["src/ttta/methods/LDA/lda_gibbs.pyx"],
                 include_dirs=[numpy.get_include()]),
                 Extension("ttta.methods.LDA.flda_c",
                           sources=["src/ttta/methods/LDA/flda_c.pyx"],
                           include_dirs=[numpy.get_include()]),
                 ],
)

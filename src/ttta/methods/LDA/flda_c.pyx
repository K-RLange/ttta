cimport numpy as np
np.import_array()

cdef extern from "flda_gibbs.h":
    double digamma(double x)

def digamma_func(double x):
    return digamma(x)
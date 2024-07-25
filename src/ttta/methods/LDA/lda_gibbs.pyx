cimport numpy as np
np.import_array()

cdef extern from "stdint.h":
    ctypedef uintc_t
    ctypedef uint_t

cdef extern from "vanilla_gibbs.h":
    void vanilla_gibbs(unsigned long long * w_vec,
                       unsigned int * as_vec,
                       unsigned long long * d_vec,
                       unsigned long long * vk_mat,
                       unsigned long long * dk_mat,
                       unsigned long long * v_sum,
                       double * alpha,
                       double * gamma,
                       long double * probs,
                       int K,
                       long V,
                       long D,
                       long W,
                       int iter,
                       int start)




def vanilla_gibbs_func(np.ndarray[np.npy_uint64, ndim=1, mode="c"] w_vec,
                  np.ndarray[np.uint32_t, ndim=1, mode="c"] as_vec,
                  np.ndarray[np.npy_uint64, ndim=1, mode="c"] d_vec,
                  np.ndarray[np.npy_uint64, ndim=2, mode="c"] vk_mat,
                  np.ndarray[np.npy_uint64, ndim=2, mode="c"] dk_mat,
                  np.ndarray[np.npy_uint64, ndim=1, mode="c"] v_sum,
                  np.ndarray[double, ndim=1, mode="c"] alpha,
                  np.ndarray[double, ndim=1, mode="c"] gamma,
                  np.ndarray[double, ndim=1, mode="c"] probs,
                  K, iter, start):

    vanilla_gibbs(<unsigned long long*> np.PyArray_DATA(w_vec),
                  <unsigned int*> np.PyArray_DATA(as_vec),
                  <unsigned long long*> np.PyArray_DATA(d_vec),
                  <unsigned long long*> np.PyArray_DATA(vk_mat),
                  <unsigned long long*> np.PyArray_DATA(dk_mat),
                  <unsigned long long*> np.PyArray_DATA(v_sum),
                  <double*> np.PyArray_DATA(alpha),
                  <double*> np.PyArray_DATA(gamma),
                  <long double*> np.PyArray_DATA(probs),
                  K,
                  vk_mat.shape[0],
                  dk_mat.shape[0],
                  w_vec.shape[0],iter,start)

cdef extern from "vanilla_gibbs.h":
    void final_assigment(unsigned long long * w_vec,
                         unsigned int * as_vec,
                         unsigned long long * d_vec,
                         unsigned long long * vk_mat,
                         unsigned long long * dk_mat,
                         unsigned long long * v_sum,
                         double * alpha,
                         double * gamma,
                         double * probs,
                         int K,
                         long V,
                         long D,
                         long W,
                         int iter,
                         long start)




def final_assignment_func(np.ndarray[np.npy_uint64, ndim=1, mode="c"] w_vec not None,
                          np.ndarray[np.uint32_t, ndim=1, mode="c"] as_vec not None,
                          np.ndarray[np.npy_uint64, ndim=1, mode="c"] d_vec not None,
                          np.ndarray[np.npy_uint64, ndim=2, mode="c"] vk_mat not None,
                          np.ndarray[np.npy_uint64, ndim=2, mode="c"] dk_mat not None,
                          np.ndarray[np.npy_uint64, ndim=1, mode="c"] v_sum not None,
                          np.ndarray[double, ndim=1, mode="c"] alpha not None,
                          np.ndarray[double, ndim=1, mode="c"] gamma not None,
                          K,iter,start):

    final_assigment(<unsigned long long*> np.PyArray_DATA(w_vec),
                    <unsigned int*> np.PyArray_DATA(as_vec),
                    <unsigned long long*> np.PyArray_DATA(d_vec),
                    <unsigned long long*> np.PyArray_DATA(vk_mat),
                    <unsigned long long*> np.PyArray_DATA(dk_mat),
                    <unsigned long long*> np.PyArray_DATA(v_sum),
                    <double*> np.PyArray_DATA(alpha),
                    <double*> np.PyArray_DATA(gamma),
                    <double*> np.PyArray_DATA(gamma),
                    K,
                    vk_mat.shape[0],
                    dk_mat.shape[0],
                    w_vec.shape[0],iter,start)



cdef extern from "vanilla_gibbs.h":
    void load_wk_mat(unsigned long long * w_vec,
                    unsigned int * as_vec,
                    unsigned long long * vk_mat,
                    int K,
                    long V,
                    long W)




def load_wk_mat_func(np.ndarray[np.npy_uint64, ndim=1, mode="c"] w_vec not None,
                     np.ndarray[np.uint32_t, ndim=1, mode="c"] as_vec not None,
                     np.ndarray[np.npy_uint64, ndim=2, mode="c"] vk_mat not None,
                     K):

    load_wk_mat(<unsigned long long*> np.PyArray_DATA(w_vec),
                <unsigned int*> np.PyArray_DATA(as_vec),
                <unsigned long long*> np.PyArray_DATA(vk_mat),
                K,
                vk_mat.shape[0],
                w_vec.shape[0])


cdef extern from "vanilla_gibbs.h":
    void load_dk_mat(unsigned long long * d_vec,
                     unsigned int * as_vec,
                     unsigned long long * dk_mat,
                     int K,
                     long D)




def load_dk_mat_func(np.ndarray[np.npy_uint64, ndim=1, mode="c"] d_vec not None,
                     np.ndarray[np.uint32_t, ndim=1, mode="c"] as_vec not None,
                     np.ndarray[np.npy_uint64, ndim=2, mode="c"] dk_mat not None,
                     K):
    load_dk_mat(<unsigned long long *> np.PyArray_DATA(d_vec),
                <unsigned int *> np.PyArray_DATA(as_vec),
                <unsigned long long *> np.PyArray_DATA(dk_mat),
                K,
                d_vec.shape[0])

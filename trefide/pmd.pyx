# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False

import sys
import os
import time
import multiprocessing

import numpy as np
cimport numpy as np

from cython.parallel import parallel, prange
from libc.stdlib cimport abort, calloc, malloc, free
from sklearn.utils.extmath import randomized_svd as svd
from functools import partial

FOV_BHEIGHT_WARNING = "Input FOV height must be an evenly divisible by block height."
FOV_BWIDTH_WARNING = "Input FOV width must be evenly divisible by block width." 
DSUB_BHEIGHT_WARNING = "Block height must be evenly divisible by spatial downsampling factor."
DSUB_BWIDTH_WARNING = "Block width must be evenly divisible by spatial downsampling factor."
TSUB_FRAMES_WARNING = "Num Frames must be evenly divisible by temporal downsampling factor."


# -----------------------------------------------------------------------------#
# ------------------------- Imports From Libtrefide.so ------------------------#
# -----------------------------------------------------------------------------#


cdef extern from "trefide.h":
    
    size_t pmd(const int d1, 
               const int d2, 
               const int nchan, 
               int d_sub, 
               const int t,
               int t_sub,
               double* R, 
               double* R_ds,
               double* U,
               double* V,
               const double spatial_thresh,
               const double temporal_thresh,
               const size_t max_components,
               const size_t consec_failures,
               const int max_iters_main,
               const int max_iters_init,
               const double tol) nogil

    void batch_pmd(const int bheight,
                   const int bwidth, 
                   const int nchan, 
                   int d_sub,
                   const int t,
                   int t_sub,
                   const int b,
                   double** Rpt, 
                   double** Rpt_ds, 
                   double** Upt,
                   double** Vpt,
                   size_t* Kpt,
                   const double spatial_thresh,
                   const double temporal_thresh,
                   const size_t max_components,
                   const size_t consec_failures,
                   const size_t max_iters_main,
                   const size_t max_iters_init,
                   const double tol) nogil

    void downsample_3d(const int d1, 
                       const int d2, 
                       const int d_sub, 
                       const int t, 
                       const int t_sub, 
                       const double *Y, 
                       double *Y_ds) nogil

# -----------------------------------------------------------------------------#
# -------------------------- Single-Block Wrapper -----------------------------#
# -----------------------------------------------------------------------------#


cpdef size_t decompose(const int d1, 
                       const int d2, 
                       const int nchan,
                       const int t,
                       double[::1] Y, 
                       double[::1] U,
                       double[::1] V,
                       const double spatial_thresh,
                       const double temporal_thresh,
                       const size_t max_components,
                       const size_t consec_failures,
                       const size_t max_iters_main,
                       const size_t max_iters_init,
                       const double tol) nogil:
    """ Wrap the single patch cpp PMD functions """

    # Turn Off Gil To Take Advantage Of Multithreaded MKL Libs
    with nogil:
        return pmd(d1, d2, nchan, 1, t, 1, &Y[0], NULL, &U[0], &V[0], 
                   spatial_thresh, temporal_thresh,
                   max_components, consec_failures, 
                   max_iters_main, max_iters_init, tol)


cpdef size_t decimated_decompose(const int d1, 
                                 const int d2,
                                 const int nchan,
                                 int d_sub,
                                 const int t,
                                 int t_sub,
                                 double[::1] Y, 
                                 double[::1] Y_ds, 
                                 double[::1] U,
                                 double[::1] V,
                                 const double spatial_thresh,
                                 const double temporal_thresh,
                                 const size_t max_components,
                                 const size_t consec_failures,
                                 const int max_iters_main,
                                 const int max_iters_init,
                                 const double tol) nogil:
    """ Wrap the single patch cpp PMD functions """

    # Turn Off Gil To Take Advantage Of Multithreaded MKL Libs
    with nogil:
        return pmd(d1, d2, nchan, d_sub, t, t_sub, &Y[0], &Y_ds[0], &U[0], &V[0], 
                   spatial_thresh, temporal_thresh,
                   max_components, consec_failures, 
                   max_iters_main, max_iters_init, tol);

# -----------------------------------------------------------------------------#
# --------------------------- Multi-Block Wrappers ----------------------------#
# -----------------------------------------------------------------------------#


cpdef batch_decompose(const int d1, 
                      const int d2, 
                      const int nchan,
                      const int t,
                      double[:, :, :,::1] Y, 
                      const int bheight,
                      const int bwidth,
                      const double spatial_thresh,
                      const double temporal_thresh,
                      const size_t max_components,
                      const size_t consec_failures,
                      const size_t max_iters_main,
                      const size_t max_iters_init,
                      const double tol,
                      int d_sub = 1,
                      int t_sub = 1):
    """ Wrapper for the .cpp parallel_factor_patch which wraps the .cpp function 
     factor_patch with OpenMP directives to parallelize batch processing."""

    # Assert Evenly Divisible FOV/Block Dimensions
    assert d1 % bheight == 0 , FOV_BHEIGHT_WARNING
    assert d2 % bwidth == 0 , FOV_BWIDTH_WARNING
    assert bheight % d_sub == 0 , DSUB_BHEIGHT_WARNING
    assert bwidth % d_sub == 0 , DSUB_BWIDTH_WARNING
    assert t % t_sub == 0 , TSUB_FRAMES_WARNING    
    
    # Initialize Counters
    cdef size_t iu, ku
    cdef int i, j, k, b, bi, bj, c
    cdef int nbi = int(d1/bheight)
    cdef int nbj = int(d2/bwidth)
    cdef int num_blocks = nbi * nbj
    cdef int bheight_ds = bheight / d_sub
    cdef int bwidth_ds = bwidth / d_sub
    cdef int t_ds = t / t_sub

    # Compute block-start indices and spatial cutoff
    indices = np.transpose([np.tile(range(nbi), nbj), np.repeat(range(nbj), nbi)])

    # Preallocate Space For Outputs
    cdef double[:,::1] U = np.zeros((num_blocks, bheight * bwidth * nchan * max_components), dtype=np.float64)
    cdef double[:,::1] V = np.zeros((num_blocks, t * max_components), dtype=np.float64)
    cdef size_t[::1] K = np.empty((num_blocks,), dtype=np.uint64)

    # Allocate Input Pointers
    cdef double** Rp = <double **> malloc(num_blocks * sizeof(double*))
    cdef double** Rp_ds = <double **> malloc(num_blocks * sizeof(double*))
    cdef double** Vp = <double **> malloc(num_blocks * sizeof(double*))
    cdef double** Up = <double **> malloc(num_blocks * sizeof(double*))

    # Release Gil Prior To Referencing Address & Calling Multithreaded Code
    with nogil:

        # Assign Pre-allocated Output Memory To Pointer Array & Allocate Residual Pointers
        for b in range(num_blocks):
            Rp[b] = <double *> malloc(bheight * bwidth * nchan * t * sizeof(double))
            Up[b] = &U[b,0]
            Vp[b] = &V[b,0] 

        # Copy Contents Of Raw Blocks Into Residual Pointers
        for bj in range(nbj):
            for bi in range(nbi):
                for k in range(t):
                    for c in range(nchan):
                        for j in range(bwidth):
                            for i in range(bheight):
                                Rp[bi + (bj * nbi)][i + (j * bheight) + (c * bheight * bwidth) + (k * nchan * bheight * bwidth)] =\
                                        Y[(bi * bheight) + i, (bj * bwidth) + j, c, k]
        
        # Decimate Raw Blocks
        if t_sub > 1 or d_sub > 1: 
            for b in range(num_blocks):
                Rp_ds[b] = <double *> malloc((bheight / d_sub) * (bwidth / d_sub) * nchan * (t / t_sub) * sizeof(double))
            for b in prange(num_blocks, schedule='guided'):
                # TODO: Generalize So downsampling occurs within each channel
                downsample_3d(bheight, bwidth, d_sub, t, t_sub, Rp[b], Rp_ds[b]) 
        else:
            for b in range(num_blocks):
                Rp_ds[b] = NULL

        # Factor Blocks In Parallel
        batch_pmd(bheight, bwidth, nchan, d_sub, t, t_sub, num_blocks, 
                  Rp, Rp_ds, Up, Vp, &K[0], 
                  spatial_thresh, temporal_thresh,
                  max_components, consec_failures, 
                  max_iters_main, max_iters_init, tol)
        
        # Free Allocated Memory
        for b in range(num_blocks):
            free(Rp[b])
        if t_sub > 1 or d_sub > 1: 
            for b in range(num_blocks):
                free(Rp_ds[b])
        free(Rp)
        free(Rp_ds)
        free(Up)
        free(Vp)
            
    # Format Components & Return To Numpy Array
    return (np.asarray(U).reshape((num_blocks, bheight, bwidth, nchan, max_components), order='F'), 
            np.asarray(V).reshape((num_blocks, max_components, t), order='C'), 
            np.asarray(K), indices.astype(np.uint64))
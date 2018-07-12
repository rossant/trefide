# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False

import os
import time
import multiprocessing

import numpy as np
import scipy.sparse as sparse
cimport numpy as np

from cython.parallel import parallel, prange
from libc.stdlib cimport malloc, free
from libcpp cimport bool
from sklearn.utils.extmath import randomized_svd as svd
from functools import partial

FOV_BHEIGHT_WARNING = "Input FOV height must be an evenly divisible by block height."
FOV_BWIDTH_WARNING = "Input FOV width must be evenly divisible by block width." 
DSUB_BHEIGHT_WARNING = "Block height must be evenly divisible by spatial downsampling factor."
DSUB_BWIDTH_WARNING = "Block width must be evenly divisible by spatial downsampling factor."
TSUB_FRAMES_WARNING = "Num Frames must be evenly divisible by temporal downsampling factor."
EVEN_BHEIGHT_WARNING = "Block height must be an even integer."
EVEN_BWIDTH_WARNING = "Block width must be an even integer."


# -----------------------------------------------------------------------------#
# ------------------------- Imports From Libtrefide.so ------------------------#
# -----------------------------------------------------------------------------#


cdef extern from "trefide.h":
    
    size_t pmd(const int d1, 
               const int d2, 
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

    void batch_pmd(const double* input_mov,
                   const size_t fov_height,
                   const size_t fov_width,
                   const size_t num_frames, 
                   const size_t num_blocks, 
                   const size_t* block_dims, 
                   const size_t* block_indices, 
                   size_t* block_ranks, 
                   double** spatial_pointers, 
                   double** temporal_pointers,
                   size_t d_sub, 
                   size_t t_sub, 
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

    double spatial_test_statistic(const int d1,
                                  const int d2,
                                  const double* u_k) nogil

    double temporal_test_statistic(const int t,
                                   const double* v_k) nogil

# -----------------------------------------------------------------------------#
# -------------------------- Single-Block Wrapper -----------------------------#
# -----------------------------------------------------------------------------#


cpdef size_t decompose(const int d1, 
                       const int d2, 
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
        return pmd(d1, d2, 1, t, 1, &Y[0], NULL, &U[0], &V[0], 
                   spatial_thresh, temporal_thresh,
                   max_components, consec_failures, 
                   max_iters_main, max_iters_init, tol)


cpdef size_t decimated_decompose(const int d1, 
                                 const int d2, 
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
        return pmd(d1, d2, d_sub, t, t_sub, &Y[0], &Y_ds[0], &U[0], &V[0], 
                   spatial_thresh, temporal_thresh,
                   max_components, consec_failures, 
                   max_iters_main, max_iters_init, tol);

# -----------------------------------------------------------------------------#
# -------------------------- Multi-Block Wrappers -----------------------------#
# -----------------------------------------------------------------------------#

# --------------------------- Block Partitioning ------------------------------#


cdef void fill_partition_metadata(const int d1, 
                                  const int d2, 
                                  const int block_height,
                                  const int block_width,
                                  const bool overlapping,
                                  size_t* block_dims,
                                  size_t* block_indices) nogil:
    """ Compute block shapes & locs for a (potentially overlapping) tiling """
    
    # Declare local variables
    cdef size_t i, j, nbi, nbj, block_index
    
    # Compute ratios of FOV dims / block dims
    nbi = d1 / block_height 
    nbj = d2 / block_width
    
    # Fill normal tiling 
    block_index = 0
    for j in range(nbj):
        
        # Full Blocks
        for i in range(nbi):
            block_indices[block_index * 2] = i * block_height
            block_indices[block_index * 2 + 1] = j * block_width 
            block_dims[block_index * 2] = block_height
            block_dims[block_index * 2 + 1] = block_width
            block_index += 1

    # Non-overlapping portion completed
    if not overlapping:
        return

    # Fill horizontally skewed tiling
    for j in range(nbj + 1):

        # Left col of half-width blocks
        if j == 0:
            for i in range(nbi):
                block_indices[block_index * 2] = i * block_height
                block_indices[block_index * 2 + 1] = 0 
                block_dims[block_index * 2] = block_height
                block_dims[block_index * 2 + 1] = block_width / 2
                block_index += 1
        
        # Middle cols of full-size blocks
        elif j < nbj:
            for i in range(nbi):
                block_indices[block_index * 2] = i * block_height
                block_indices[block_index * 2 + 1] = j * block_width - block_width / 2
                block_dims[block_index * 2] = block_height
                block_dims[block_index * 2 + 1] = block_width
                block_index += 1

        # Right col of half-width blocks
        else:
            for i in range(nbi):
                block_indices[block_index * 2] = i * block_height
                block_indices[block_index * 2 + 1] = d2 - block_width / 2 
                block_dims[block_index * 2] = block_height
                block_dims[block_index * 2 + 1] = block_width / 2
                block_index += 1

    # Fill vertically skewed tiling
    for j in range(nbj):

        # Top row of half-height blocks
        block_indices[block_index * 2] = 0
        block_indices[block_index * 2 + 1] = j * block_width 
        block_dims[block_index * 2] = block_height / 2
        block_dims[block_index * 2 + 1] = block_width
        block_index += 1

        # Middle rows of full-size blocks
        for i in range(nbi -1):
            block_indices[block_index * 2] = block_height / 2 + i * block_height
            block_indices[block_index * 2 + 1] = j * block_width 
            block_dims[block_index * 2] = block_height
            block_dims[block_index * 2 + 1] = block_width
            block_index += 1

        # Bottom row of half-height blocks
        block_indices[block_index * 2] = d1 - block_height / 2
        block_indices[block_index * 2 + 1] = j * block_width 
        block_dims[block_index * 2] = block_height / 2
        block_dims[block_index * 2 + 1] = block_width
        block_index += 1


    # Fill diagonally skewed tiling
    for j in range(nbj + 1):

        # Left col of half-width blocks
        if j == 0:
            
            # Top left corner
            block_indices[block_index * 2] = 0
            block_indices[block_index * 2 + 1] = 0
            block_dims[block_index * 2] = block_height / 2
            block_dims[block_index * 2 + 1] = block_width / 2
            block_index += 1

            # Middle rows of full-height blocks
            for i in range(nbi - 1):
                block_indices[block_index * 2] = block_height / 2 + i * block_height
                block_indices[block_index * 2 + 1] = 0
                block_dims[block_index * 2] = block_height
                block_dims[block_index * 2 + 1] = block_width / 2
                block_index += 1
            
            # Bottom left corner
            block_indices[block_index * 2] = d1 - block_height / 2
            block_indices[block_index * 2 + 1] = 0
            block_dims[block_index * 2] = block_height / 2
            block_dims[block_index * 2 + 1] = block_width / 2
            block_index += 1

        # Middle cols of full-width blocks
        elif j < nbj:

            # Top row of half-height blocks
            block_indices[block_index * 2] = 0
            block_indices[block_index * 2 + 1] = j * block_width - block_width / 2
            block_dims[block_index * 2] = block_height / 2
            block_dims[block_index * 2 + 1] = block_width
            block_index += 1

            # Middle rows of full-size blocks
            for i in range(nbi - 1):
                block_indices[block_index * 2] = block_height / 2 + i * block_height
                block_indices[block_index * 2 + 1] = j * block_width - block_width / 2
                block_dims[block_index * 2] = block_height
                block_dims[block_index * 2 + 1] = block_width
                block_index += 1
            
            # Bottom row of half-height blocks
            block_indices[block_index * 2] = d1 - block_height / 2
            block_indices[block_index * 2 + 1] = j * block_width - block_width / 2
            block_dims[block_index * 2] = block_height / 2
            block_dims[block_index * 2 + 1] = block_width
            block_index += 1

        # Right col of half-width blocks
        else:
            
            # Top right corner
            block_indices[block_index * 2] = 0
            block_indices[block_index * 2 + 1] = d2 - block_width / 2
            block_dims[block_index * 2] = block_height / 2
            block_dims[block_index * 2 + 1] = block_width / 2
            block_index += 1

            # Middle rows of full-height blocks
            for i in range(nbi - 1):
                block_indices[block_index * 2] = block_height / 2 + i * block_height
                block_indices[block_index * 2 + 1] = d2 - block_width / 2
                block_dims[block_index * 2] = block_height
                block_dims[block_index * 2 + 1] = block_width / 2
                block_index += 1

            # Bottom right corner
            block_indices[block_index * 2] = d1 - block_height / 2
            block_indices[block_index * 2 + 1] = d2 - block_width / 2
            block_dims[block_index * 2] = block_height / 2
            block_dims[block_index * 2 + 1] = block_width / 2
            block_index += 1

# ------------------------ Compressed Component Formatting ------------------------#


cpdef size_t[::1] get_format_metadata(const size_t[::1] block_ranks, 
                                      const size_t[::1] block_dims,
                                      const size_t num_frames,
                                      const size_t num_blocks):
    """ Compute access indices for spatial components stored in compressed format """

    # Allocate Internal Vars
    cdef size_t bdx, num_elems

    # Compute starting indices of each block in compressed format
    cdef size_t[::1] block_access = np.zeros((num_blocks + 1) * 2, dtype=np.uint64)
    for bdx in range(num_blocks):

        # Spatial blocks: pixels vary by block
        num_elems = block_dims[bdx * 2] * block_dims[bdx * 2 + 1] * block_ranks[bdx]
        block_access[bdx * 2 + 2] = block_access[bdx * 2] + num_elems

        # Temporal block: frames constant across blocks
        block_access[bdx * 2 + 3] = block_access[bdx * 2 + 1] + num_frames * block_ranks[bdx]

    # Return array of spatial block starting indices
    return block_access


cdef double[::1] format_spatial(const size_t[::1] block_ranks,
                                const size_t[::1] block_dims,
                                const size_t num_blocks,
                                const size_t[::1] block_access,
                                const double[:,:] spatial_components):
    """ Compactly reformat the spatial components as a single array """

    # Allocate Space For Output Array
    cdef double[::1] spatial_outputs = np.zeros(block_access[num_blocks * 2], dtype=np.float64)

    # Copy Data For Used Components
    cdef size_t bdx, num_elems
    for bdx in range(num_blocks):
        num_elems = block_dims[bdx * 2] * block_dims[bdx * 2 + 1] * block_ranks[bdx]
        spatial_outputs[block_access[bdx * 2]:block_access[bdx * 2] + num_elems] = spatial_components[bdx, :num_elems] 

    # Return single array containing all spatial components
    return spatial_outputs


cdef double[::1] format_temporal(const size_t[::1] block_ranks,
                                 const size_t num_frames,
                                 const size_t num_blocks,
                                 const size_t[::1] block_access,
                                 const double[:,::1] temporal_components):
    """ Compactly reformat the temporal components as a single array """

    # Allocate Space For Output Array 
    cdef double[::1] temporal_outputs = np.zeros(block_access[num_blocks * 2 + 1], dtype=np.float64)

    # Copy Data For Used Components
    cdef size_t bdx, num_elems
    for bdx in range(num_blocks):
        num_elems = num_frames * block_ranks[bdx]
        temporal_outputs[block_access[bdx * 2 + 1]:block_access[bdx * 2 + 1] + num_elems] = temporal_components[bdx, :num_elems] 

    # Return single array containing all temporal components
    return temporal_outputs


cpdef size_t compression_factor(const size_t[::1] block_ranks,
                                const size_t[::1] block_dims,
                                const size_t fov_height,
                                const size_t fov_width,
                                const size_t num_frames,
                                const size_t num_blocks):
    """ Compute the compressin factor achieved by storing movie as factored 
       components as compared to naive representation as full matrix. """

    # Compute Size Of Compressed Factors
    cdef size_t bdx, compressed_size = 0
    for bdx in range(num_blocks):
        compressed_size += block_ranks[bdx] * (block_dims[bdx*2] * block_dims[bdx*2+1] + num_frames)

    # Take Ratio With Size Of Original Mov Matrix
    return (fov_height * fov_width * num_frames) / compressed_size


# ------------------------ Full FOV PMD(TF,TV) Wrappers ------------------------#


cpdef blockwise_decompose(const double[:, :, ::1] input_mov, 
                          const size_t fov_height, 
                          const size_t fov_width, 
                          const size_t num_frames,
                          const size_t block_height,
                          const size_t block_width,
                          const double spatial_thresh,
                          const double temporal_thresh,
                          const size_t max_components,
                          const size_t consec_failures,
                          const size_t max_iters_main,
                          const size_t max_iters_init,
                          const double tol,
                          bool overlapping=True,
                          size_t d_sub=1,
                          size_t t_sub=1):
    """ Perform blockwise PMD(TV,TF) decomposition on an input movie Y """
    
    # Validate Input Parameter Choices
    assert fov_height % block_height == 0, FOV_BHEIGHT_WARNING
    assert fov_width % block_width == 0, FOV_BWIDTH_WARNING
    assert block_height % d_sub == 0, DSUB_BHEIGHT_WARNING
    assert block_width % d_sub == 0, DSUB_BWIDTH_WARNING
    assert num_frames % t_sub == 0, TSUB_FRAMES_WARNING    
    if overlapping:
        assert block_height % 2 == 0, EVEN_BHEIGHT_WARNING
        assert block_width % 2 == 0, EVEN_BWIDTH_WARNING
     
    # Determine number of blocks in (potentially overlapping) partition
    cdef size_t num_block_rows = fov_height / block_height
    cdef size_t num_block_cols = fov_width / block_width
    cdef size_t num_blocks
    if overlapping:
        num_blocks = (2 * num_block_rows + 1) * (2 * num_block_cols + 1)
    else:
        num_blocks = num_block_rows * num_block_cols

    # Prepare block partition metadata
    cdef size_t[::1] block_dims = np.zeros((num_blocks * 2), dtype=np.uint64)
    cdef size_t[::1] block_indices = np.zeros((num_blocks * 2), dtype=np.uint64) 
    cdef size_t[::1] block_ranks = np.zeros((num_blocks,), dtype=np.uint64)
    fill_partition_metadata(fov_height, fov_width, block_height, block_width,
                            overlapping, &block_dims[0], &block_indices[0])
    
    # Allocate space for spatial components 
    cdef double[:,::1] spatial_components = np.zeros(
            (num_blocks, block_height * block_width * max_components), 
            dtype=np.float64
        )
    cdef double** spatial_pointers = <double **> malloc(num_blocks  * sizeof(double*))
    
    # Allocate space for temporal components
    cdef double[:,::1] temporal_components = np.zeros(
            (num_blocks, num_frames * max_components),
            dtype=np.float64
        )
    cdef double** temporal_pointers = <double **> malloc(num_blocks * sizeof(double*))
    
    # Assign pointers to component ndarrays
    cdef size_t bdx
    for bdx in range(num_blocks):
        spatial_pointers[bdx] = &spatial_components[bdx, 0]
        temporal_pointers[bdx] = &temporal_components[bdx, 0]
    
    # Factor Blocks In Parallel
    batch_pmd(&input_mov[0,0,0], fov_height, fov_width, num_frames, num_blocks, 
              &block_dims[0], &block_indices[0], &block_ranks[0], 
              spatial_pointers, temporal_pointers, d_sub, t_sub, 
              spatial_thresh, temporal_thresh,
              max_components, consec_failures, 
              max_iters_main, max_iters_init, tol)

    # Free Allocated Memory
    free(spatial_pointers)
    free(temporal_pointers)
            
    # Reformat components to store with (minimum amount of) contiguous memory
    cdef size_t[::1] block_access = get_format_metadata(block_ranks, block_dims, num_frames, num_blocks)
    cdef double[::1] spatial_outputs = format_spatial(block_ranks, block_dims, num_blocks, block_access, spatial_components)
    cdef double[::1] temporal_outputs = format_temporal(block_ranks, num_frames, num_blocks, block_access, temporal_components) 

    # Reformat (input & output) metadata into nested dictionary
    metadata = {'params':{'fov_height': fov_height,
                          'fov_width': fov_width,
                          'num_frames': num_frames,
                          'block_height': block_height,
                          'block_width': block_width,
                          'spatial_thresh': spatial_thresh,
                          'temporal_thresh': temporal_thresh,
                          'max_components': max_components,
                          'consec_failures': consec_failures,
                          'max_iters_main': max_iters_main,
                          'max_iters_init': max_iters_init,
                          'tol': tol,
                          'overlapping': overlapping,
                          'd_sub': d_sub,
                          't_sub': t_sub},
                'format':{'num_blocks': num_blocks,
                          'total_rank': block_access[num_blocks *2 + 1] / num_frames,
                          'fov_indices': block_indices,
                          'format_indices': block_access,
                          'dims': block_dims,
                          'ranks': block_ranks,
                          'compression_factor': compression_factor(block_ranks,
                                                                   block_dims,
                                                                   fov_height,
                                                                   fov_width,
                                                                   num_frames,
                                                                   num_blocks)}}

    return spatial_outputs, temporal_outputs, metadata


# -----------------------------------------------------------------------------#
# ---------------------- Compression Format Utilities -------------------------#
# -----------------------------------------------------------------------------#

# --- Overlapping Recombination Weights ---


cpdef double[::1,:] get_quadrant_weights(size_t block_height, 
                                         size_t block_width): 
    """ Populate The Upper Left Corner Of The Recombination Weight Matrix """

    # Declare Local Variables
    cdef size_t i, j

    # Populate Upper Left Corner Of Weighting Matrix
    cdef double[::1,:] ul_corner = np.empty((block_height / 2, block_width / 2),
                                            dtype=np.float64, order='F')
    for j in range(block_width / 2):
        for i in range(block_height / 2):
            ul_corner[i,j] = min(i, j)

    # Compute & Normalize By Cumulative Overlapped Weights
    cdef double[::1,:] cum_weights = np.asfortranarray(np.asarray(ul_corner) +\
            np.fliplr(ul_corner) + np.flipud(ul_corner) +\
            np.fliplr(np.flipud(ul_corner))) 
    for j in range(block_width / 2):
        for i in range(block_height / 2):
            ul_corner[i,j] = ul_corner[i,j] / cum_weights[i,j]

    return ul_corner


cdef double* get_weight_pointer(double* full, double* top_half, double* bottom_half,
                                size_t block_height, size_t block_width,
                                bool vert_small, bool horiz_small, 
                                bool left_edge, bool top_edge) nogil:
    """ Find Starting Memory Location From Which To Apply Weights"""

    if vert_small and horiz_small:  # Corners

        if left_edge and top_edge:  # Upper Left
            return &bottom_half[block_height * block_width]

        elif left_edge:  # Lower Left 
            return &top_half[block_height * block_width] 

        elif top_edge:  # Upper Right
            return &bottom_half[0]

        else:  # Lower Right
            return &top_half[0]
    
    elif vert_small:  # Top & Bottom Rows
        
        if top_edge:  # Top
            return &bottom_half[0]

        else:  #  Bottom
            return &top_half[0]
    
    elif horiz_small:  # Left & Right Edges
        
        if left_edge:  # Left
            return &full[block_height * block_width]
        
        else:  # Right
            return &full[0]
    
    else:  # Standard Blocks
        return &full[0]


cdef void apply_weights(const double* weights, 
                        const size_t block_size,
                        double* spatial_component) nogil:
    """ Modify A Spatial Component Inplace By Applying Recombination Wewights """
    cdef size_t idx
    for idx in range(block_size):
        spatial_component[idx] *= weights[idx]


cpdef void apply_recombination_weights(dict metadata, 
                                       double[::1] spatial_components):
    """ Updates Spatial Components Inplace By Applying Overlapping Recombination Weights"""
    
    # Extract Relevant Metadata
    cdef size_t[::1] block_indices = metadata['format']['fov_indices']
    cdef size_t[::1] block_dims = metadata['format']['dims']
    cdef size_t[::1] block_ranks = metadata['format']['ranks']
    cdef size_t[::1] block_access = metadata['format']['format_indices']
    cdef size_t num_blocks = metadata['format']['num_blocks']
    cdef size_t block_height = metadata['params']['block_height']
    cdef size_t block_width = metadata['params']['block_width'] 

    # Compute Recombination Matrix For Standard Block
    cdef double[::1,:] ul_corner = get_quadrant_weights(block_height, block_width)
    cdef double[::1,:] top_half = np.asfortranarray(np.hstack([ul_corner, np.fliplr(ul_corner)]))
    cdef double[::1,:] bottom_half = np.asfortranarray(np.flipud(top_half))
    cdef double[::1,:] full = np.asfortranarray(np.vstack([top_half, bottom_half]))

    # Cycle Through Blocks
    cdef size_t bdx, block_size, curr_index, k
    cdef bool vert_small, horiz_small
    cdef double* weight_pointer
    for bdx in range(num_blocks):
        
        # Check For Literal Edge & Corner Cases
        vert_small = block_dims[bdx * 2] < block_height
        horiz_small = block_dims[bdx * 2 + 1] < block_width
        left_edge = block_indices[bdx * 2] == 0
        top_edge = block_indices[bdx * 2 + 1] == 0
        
        # Get Memory Address Of Weights For This Block
        weight_pointer = get_weight_pointer(&full[0,0], &top_half[0,0], &bottom_half[0,0],
                                            block_dims[bdx * 2], block_dims[bdx * 2 + 1], 
                                            vert_small, horiz_small, 
                                            left_edge, top_edge)

        # Apply Weights To Each Spatial Component In Block
        block_size = block_dims[bdx * 2] * block_dims[bdx * 2 + 1]
        for k in range(block_ranks[bdx]):
            curr_index = block_access[bdx * 2] + k * block_size
            apply_weights(weight_pointer, block_size, &spatial_components[curr_index])
 

# --- Reformatting ---


cpdef format_as_matrices(double[::1] spatial_components, 
                         double[::1] temporal_components, 
                         dict metadata):
    """ Reformat the compressed spatial and temporal components into the 
        structured matrices U, V where U is sparse and V is dense"""

    # Declare Internal Variables
    cdef size_t bdx, k, j0, j, i0, i, block_height, block_width

    # Unpack Useful Params From Metadata
    cdef size_t fov_height = metadata['params']['fov_height']
    cdef size_t fov_width = metadata['params']['fov_width']
    cdef size_t num_frames = metadata['params']['num_frames']
    cdef size_t num_blocks = metadata['format']['num_blocks']
    cdef size_t total_rank = metadata['format']['total_rank']
    cdef size_t[::1] block_indices = metadata['format']['fov_indices']
    cdef size_t[::1] block_access = metadata['format']['format_indices']
    cdef size_t[::1] block_dims = metadata['format']['dims']
    cdef size_t[::1] block_ranks = metadata['format']['ranks']
 
    # Allocate Space For Sparse Matrix Representation
    cdef size_t[::1] indices = np.zeros(block_access[num_blocks * 2], dtype=np.uint64)
    cdef size_t[::1] indptr = np.zeros(total_rank + 1, dtype=np.uint64)
    cdef size_t rank_ind = 0
    for bdx in range(num_blocks):
        block_height = block_dims[bdx * 2]
        block_width = block_dims[bdx * 2 + 1]
        i0 = block_indices[bdx * 2]
        j0 = block_indices[bdx * 2 + 1]
        for k in range(block_ranks[bdx]):
            indptr[rank_ind+1] = indptr[rank_ind] + block_height * block_width 
            rank_ind += 1
            for j in range(block_width):
                for i in range(block_height):
                    indices[block_access[bdx * 2] + (k * block_height * block_width) + (block_height * j) + i] = fov_height * (j0 + j) + i0 + i

    # Construct Scipy CSC Format Object
    U = sparse.csc_matrix((np.array(spatial_components), np.array(indices), np.array(indptr)),
                          shape=(fov_height * fov_width, total_rank)) 
 
    # Construct Numpy Object From (Column Major) Array 
    V = np.reshape(temporal_components, (total_rank, num_frames))

    # Return structured matrices
    return U, V



# -----------------------------------------------------------------------------#
# ---------------------- Threshold Simulation Wrapper -------------------------#
# -----------------------------------------------------------------------------#


cdef void evaluate_spatial_statistics(double* spatial_components,
                                      double* spatial_stats,
                                      size_t block_height,
                                      size_t block_width,
                                      size_t num_components):
    """ """
    cdef int k
    with nogil:
        for k in prange(<int> num_components, schedule="static"):
            spatial_stats[k] = spatial_test_statistic(block_height, block_width,
                                                      &spatial_components[block_height * block_width * k])
 

cdef void evaluate_temporal_statistics(double* temporal_components,
                                       double* temporal_stats,
                                       size_t num_frames,
                                       size_t num_components):
    """ """
    cdef int k
    with nogil:
        for k in prange(<int> num_components, schedule="static"):
            temporal_stats[k] = temporal_test_statistic(num_frames, 
                                                        &temporal_components[num_components * k])


def simulate_thresholds(size_t block_height, 
                        size_t block_width, 
                        size_t num_frames, 
                        size_t num_blocks, 
                        size_t max_threads, 
                        size_t consec_failures, 
                        size_t max_iters_main, 
                        size_t max_iters_init, 
                        double tol, 
                        size_t d_sub, 
                        size_t t_sub, 
                        double conf):
    """ """

    # Compute Iterations Required To Reduce Memory Used
    cdef size_t repetitions = num_blocks / max_threads

    # Allocate Space For Simulated Statistics
    cdef double[::1] spatial_stats = np.zeros(num_blocks * consec_failures, 
                                              dtype=np.float64)
    cdef double[::1] temporal_stats = np.zeros(num_blocks * consec_failures,
                                               dtype=np.float64)

    # Allocate space for spatial components 
    cdef double[:,::1] spatial_components = np.zeros(
            (num_blocks, block_height * block_width * consec_failures), 
            dtype=np.float64
        )
    cdef double** spatial_pointers = <double **> malloc(max_threads * sizeof(double*))

    # Allocate space for temporal components
    cdef double[:,::1] temporal_components = np.zeros(
            (num_blocks, num_frames * consec_failures),
            dtype=np.float64
        )
    cdef double** temporal_pointers = <double **> malloc(max_threads * sizeof(double*))

    # Assign pointers to component ndarrays

    # Break into repetitions to avoid memory issues 
    cdef size_t bdx, rep
    cdef size_t[::1] block_dims, block_indices, block_ranks
    cdef double[:,:,::1] noise
    for rep in range(repetitions):

        # Simulate Noise
        noise = np.reshape(np.random.randn(block_height * block_width * max_threads * num_frames),
                           (block_height, block_width * max_threads, num_frames))

        # Allocate Space For Decomposition
        block_dims = np.zeros((max_threads * 2), dtype=np.uint64)
        block_indices = np.zeros((max_threads * 2), dtype=np.uint64) 
        block_ranks = np.zeros((max_threads,), dtype=np.uint64)
        fill_partition_metadata(block_height, max_threads * block_width, 
                                block_height, block_width, False,
                                &block_dims[0], &block_indices[0])

        # Transfer Components Pointers
        for bdx in range(max_threads):
            spatial_pointers[bdx] = &spatial_components[rep * max_threads + bdx, 0]
            temporal_pointers[bdx] = &temporal_components[rep * max_threads + bdx, 0]

        # Factor Blocks In Parallel
        batch_pmd(&noise[0,0,0], block_height, block_width * max_threads, num_frames,
                  max_threads, &block_dims[0], &block_indices[0], &block_ranks[0], 
                  spatial_pointers, temporal_pointers, d_sub, t_sub, 1e3, 1e3, 
                  consec_failures, consec_failures, max_iters_main, max_iters_init, tol)

    # Translate Components To Statistics
    evaluate_spatial_statistics(&spatial_components[0,0], &spatial_stats[0],
                                block_height, block_width, num_blocks * consec_failures)
    evaluate_temporal_statistics(&temporal_components[0,0], &temporal_stats[0],
                                 num_frames, num_blocks * consec_failures)

    # Compute Thresholds
    spatial_thresh =  np.percentile(spatial_stats, conf)
    temporal_thresh = np.percentile(temporal_stats, conf)

    return (spatial_thresh, temporal_thresh), spatial_stats, temporal_stats

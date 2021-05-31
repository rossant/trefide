#!/usr/bin/env python
import os
import datetime
import sys
import multiprocessing
import functools

import yaml

import numpy as np

import scipy
import scipy.sparse
import scipy.sparse.linalg

import sklearn
import skimage
import skimage.io

from sklearn.utils.extmath import randomized_svd

import trefide.pmd
from trefide.pmd import decompose
from trefide.pmd import decimated_decompose
from trefide.pmd import batch_decompose
from trefide.utils import psd_noise_estimate

# Declare Constants
CONFIG_NAME = 'config'
CONFIG_EXT = 'yaml'
REQ_PARAM_KEYS = [
    'block_height',
    'block_width'
]
DEFAULT_PARAMS = {
    # Tiff Stack Order
    'transpose': True,
    # Temporal Windowing
    'window_length': None,
    # Preprocessing: Optional
    'center': True,
    'scale': True,
    'background_rank': 0,
    # PMD Simulation: Optional
    'num_sims': 64,
    'sim_conf': 5,
    'spatial_thresh': None,
    'temporal_thresh': None,
    # PMD: Optional
    'overlapping': True,
    'd_sub': 1,
    't_sub': 1,
    'max_iters_init': 40,
    'max_iters_main': 10,
    'max_components': 25,
    'consec_failures': 3,
    'tol': 5e-3,
    # Post-Selection
    'post_selection_threshold': .98
}


def parinit():
    import os
    os.environ['MKL_NUM_THREADS'] = "1"
    os.environ['OMP_NUM_THREADS'] = "1"


def runpar(f, X, nprocesses=None, **kwargs):
    '''
    res = runpar(function,          # function to execute
                 data,              # data to be passed to the function
                 nprocesses = None, # defaults to the number of cores on the machine
                 **kwargs)          # additional arguments passed to the function (dictionary)
    '''
    if nprocesses is None:
        nprocesses = multiprocessing.cpu_count()
    with multiprocessing.Pool(initializer=parinit, processes=nprocesses) as pool:
        res = pool.map(functools.partial(f, **kwargs), X)
    pool.join()
    return res


def load_params(filename, ext):
    """ Read Config.yaml & merge with defaults """

    # Convert yaml file to python dict
    display("Loading configuration file ({})...".format(filename + '.' + ext))
    with open(filename + '.' + ext, 'r') as stream:
        user_params = yaml.safe_load(stream)
    display("Configuration file successfully loaded.")

    # Ensure That Required Arguments Have Been Provided
    display('Checking required fields...')
    for key in REQ_PARAM_KEYS:
        if not key in user_params.keys():
            raise ValueError('Provided config missing required param: {}'.format(key))
    display('All required fields have been provided.')

    # Plug In Defaults For Missing Optional Params
    display('Inserting defaults for missing optional arguments')
    for key, val in DEFAULT_PARAMS.items():
        if not key in user_params.keys():
            display('Using default {0} = {1}'.format(key, val))
            user_params[key] = val

    # Return Processed Param Dict
    return user_params


def validate_param_compatibility(params, data):
    """ Testing: ensure these hold in right conditions """

    # Get & Validate Provided Data Shape
    fov_height, fov_width, total_frames = data.shape

    # Set Data Shape Fields
    params['fov_height'] = fov_height
    params['fov_width'] = fov_width
    params['num_frames'] = total_frames

    # Set Window Length If Not Provided or Validate Provided Value
    if params['window_length'] is None:
        params['window_length'] = min(
            int(params['block_height'] * params['block_width']),
            params['num_frames']
        )
    if params['window_length'] > params['num_frames']:
        raise Exception("Window length cannot exceed number of frames")

    # FOV Dimensions Must Be Evenly Divisible Corresponding Block Dimensions
    if params['fov_height'] % params['block_height'] > 0:
        raise Exception("FOV height must be divisible by block height")
    if params['fov_width'] % params['block_width'] > 0:
        raise Exception("FOV width must be divisible by block width")

    # If Decimating, Dimensions Must Be Divisible By Downsampling Factors
    if params['block_height'] % params['d_sub'] > 0:
        raise Exception("Block height must be dividible by spatial downsampling factor")
    if params['block_width'] % params['d_sub'] > 0:
        raise Exception("Block width must be divisible by spatial downsampling factor")
    if params['window_length'] % params['t_sub'] > 0:
        raise Exception("Window length must be divisible by temporal downsampling factor")

    # Blocks Must Have Even Dimensions For Overlapping Decomposition
    # FUTURE: remove constaint with off-by-one slicing & modified weights
    if params['overlapping']:
        if params['block_height'] % 2 > 0:
            raise Exception("Block height must be even for overlapping decompositon")
        if params['block_width'] % 2 > 0:
            raise Exception("Block width must be even for overlapping decomposition")

    if params['post_selection_threshold'] <= 0 or params['post_selection_threshold'] > 1:
        raise Exception("Post selection variance threshold must be in the interval (0,1].")


def eval_spatial_stats(u):
    num_pixels, num_comps = np.prod(u.shape[:-1]), u.shape[-1]
    vert_diffs = np.abs(u[1:,:,:] - u[:-1,:,:])
    horz_diffs = np.abs(u[:,1:,:] - u[:,:-1,:])
    avg_diff = ((np.sum(vert_diffs.reshape((-1, num_comps), order='F'), axis=0) +
                 np.sum(horz_diffs.reshape((-1, num_comps), order='F'), axis=0)) /
                (np.prod(vert_diffs.shape[:-1]) + np.prod(horz_diffs.shape[:-1])))
    avg_elem = np.mean(np.abs(u.reshape((num_pixels,-1), order='F')), axis=0)
    return avg_diff / avg_elem


def eval_temporal_stats(v):
    return np.mean(np.abs(v[:, :-2] + v[:, 2:] - 2*v[:, 1:-1]), axis=-1) / np.mean(np.abs(v), axis=-1)


def noise_decomp(num_components, params=None):
    """ TESTING: downsampled case """
    E = np.random.randn(params['block_height'], params['block_width'], params['window_length'])
    U = np.zeros((params['block_height'] * params['block_width'] * num_components,), dtype=np.float64)
    V = np.zeros((params['window_length'] * num_components,), dtype=np.float64)

    if params['d_sub'] > 1 or params['t_sub'] > 1:
        # Downsample Simulated Noise
        E_sub = skimage.measure.block_reduce(E, (params['d_sub'], params['d_sub'], params['t_sub']), func=np.mean)

        # Perform Partial Decomposition Without Thresholds W/ Decimated Init
        _ = decimated_decompose(params['block_height'],
                                params['block_width'],
                                params['d_sub'],
                                params['window_length'],
                                params['t_sub'],
                                np.reshape(E, (-1,)), np.reshape(E_sub, (-1,)),
                                U, V,
                                1e3, 1e3,
                                num_components, num_components,
                                params['max_iters_main'], params['max_iters_init'],
                                params['tol'])
    else:
        # Perform Partial Decomposition Without Thresholds
        _ = decompose(params['block_height'],
                      params['block_width'],
                      params['window_length'],
                      np.reshape(E, (-1,)),
                      U, V,
                      1e3, 1e3,
                      num_components, num_components,
                      params['max_iters_main'], params['max_iters_init'],
                      params['tol'])

    # Return Summary Stats
    return (eval_spatial_stats(U.reshape((params['block_height'], params['block_width'], -1), order='F')),
            eval_temporal_stats(V.reshape((-1, params['window_length']))))


def simulate_missing_params(params):
    """ FUTURE lookup table with results shared using common precomputed thresholds """
    # Check To See If We Need To Run Simulations
    missing_spatial_thresh = params['spatial_thresh'] is None
    missing_temporal_thresh = params['temporal_thresh'] is None
    run_sims = missing_spatial_thresh or missing_temporal_thresh
    
    # If Either Is Missing, Run Simulations & Overwrite
    if run_sims:
        display("One or both thresholds were not provided, performing simulations...")

        # Each Thread Creates Noise Block & Decomposes
        results = runpar(noise_decomp, [params['consec_failures']] * params['num_sims'], params=params)

        # Aggregate Stats Across All Components From All Blocks
        agg_stats = np.array(results)
        params['spatial_thresh'] = np.percentile(agg_stats[:, 0].ravel(), params['sim_conf']).item()
        params['temporal_thresh'] = np.percentile(agg_stats[:, 1].ravel(), params['sim_conf']).item()
        display(
            "Simulations complete, determined thresholds are (spatial:{}, temporal:{})".format(
                params['spatial_thresh'], params['temporal_thresh']
            ))
    return run_sims


def write_params(filename, ext, params, thresholds_simulated=False):
    """ Read Config.yaml & merge with defaults """

    # Add Field To Keep Track Of Whether Thresholds Were Simulated Or Provided
    params['thresholds_simulated'] = thresholds_simulated

    # Convert yaml file to python dict
    display("Writing verbose configuration file ({})...".format(filename + '.' + ext))
    with open(filename + '.' + ext, 'w') as stream:
        _ = yaml.dump(params, stream, default_flow_style=False)
    display("Verbose configuration file written to outputs successfully.")


def load_data(indir, filename, ext):
    """ Handle Loading Of Multiple File Formats """

    # Switch To Different Loading Functions Depending On Filetype
    display("Loading dataset ({})...".format(filename + '.' + ext))
    if ext == 'npy':
        data = np.load(os.path.join(indir, filename) + '.' + ext)
    elif ext == 'tiff':
        data = skimage.io.imread(os.path.join(indir,filename) + '.' + ext)
    else:
        raise ValueError("Invalid file format '{}', please use ['npy', 'tiff'].")
    display("Dataset of shape {} successfully loaded.".format(data.shape))

    # Rearrange Dimensions If Instructed
    # FUTURE modify params so transpose can be any sequence of dims
    if params['transpose']:
        # FUTURE reduce bottleneck
        display("Transposing data to order (fov_height, fov_width, num_frames)...")
        data = np.transpose(data, (1, 2, 0))

    # FUTURE: add options for users to crop dimensions
    # FUTURE: reduce bottleneck
    return np.ascontiguousarray(data).astype(np.float64)


def standardize(data, center=True, scale=True):
    """ Performs inplace pixelwise standarization using mean and estimated noise stdv
        Currently requires data to be float64 C-contiguous
        FUTURE: update psd_noise_estimate code in order to handle float32 &/or uint16
        FUTURE: parallelize psd_noise_estimate over pixels
    """

    # Center With Pixelwise Median
    if center:
        display('Computing pixelwise mean...')
        baseline = np.mean(data, axis=-1)
        display('Performing inplace centering...')
        data -= baseline[:, :, None]
        display('Centering complete.')
    else:
        baseline = np.zeros((data.shape[0], data.shape[1]))

    # Scale To Have Unit Standard Deviation
    if scale:
        display('Estimating pixelwise noise variance...')
        # TODO: eliminate bottleneck
        scale = np.sqrt(
            np.reshape(psd_noise_estimate(np.reshape(data, (-1, data.shape[-1]))),
                       data.shape[:-1])
        )
        scale[scale == 0] = 1.0
        display('Performing inplace normalization...')
        data /= scale[:, :, None]
        display('Scaling complete.')
    else:
        scale = np.ones((data.shape[0], data.shape[1]))

    return data, baseline.astype(np.float32), scale.astype(np.float32)


def extract_background(params, data):
    """ """
    if params['background_rank'] > 0:
        display('Fitting low-rank background...')
        U, s, Vt = randomized_svd(
            M=data.reshape((-1, data.shape[-1])),
            n_components=params['background_rank']
        )
        Vt *= s[:, None]
        display('Removing background from dataset...')
        data -= np.dot(U, Vt).reshape(data.shape)
        display('Background successfully extracted.')
        return data, (U.astype(np.float32), Vt.astype(np.float32))
    return data, (None, None)


def residual_projection(U, Y):
    """ """
    #
    V = np.dot(
        np.linalg.pinv(np.dot(U.T, U)),
        np.dot(U.T, np.reshape(Y, (-1, Y.shape[-1])))
    )
    Y -= np.reshape(np.dot(U, V), Y.shape)
    return Y


def windowed_pmd(params, data):
    """ By default non-overlapping, can be called multiple times for overlap """

    # Allocate resid buffer by copying the first window
    residual = np.copy(data[:, :, 0:params['window_length']])
    bheight, bwidth, num_frames = params['block_height'], params['block_width'], params['num_frames']
    fov_height, fov_width, win_frames = residual.shape

    # Initialize decomp with first window
    compact_spatial, _ , block_ranks, block_coords = batch_decompose(
            fov_height, fov_width, win_frames, residual,
            params['block_height'], params['block_width'],
            params['spatial_thresh'], params['temporal_thresh'],
            params['max_components'], params['consec_failures'],
            params['max_iters_main'], params['max_iters_init'],
            params['tol'], params['d_sub'], params['t_sub']
    )

    # Sequential Rolling Window Over Remaining Frames
    for t0 in range(win_frames, params['num_frames'], win_frames):
        display('Windowed PMD: {0} / {1} frames.'.format(t0,params['num_frames']))
        # Preemptive Exit If All Blocks Are Saturated
        max_remaining_components = params['max_components'] - np.min(block_ranks)
        if max_remaining_components == 0:
            break
        
        t1 = min(t0 + win_frames, params['num_frames'])
        if t1 - t0 < win_frames:
            t0 = t1 - win_frames
        residual[:,:,:] = data[:,:,t0:t1]

        ## Update Residual Independently In Each Block
        for bdx, coords in enumerate(block_coords):

            # Nothing To Do If No Spatial Components Extracted Yet 
            rank = int(block_ranks[bdx])
            if rank > 0:
                
                # Compute indices wrt. full fov
                y0 = int(bheight * coords[0])
                x0 = int(bwidth * coords[1])
                
                # Remove Data's Projection Onto Spatial Basis 
                residual_projection(
                    np.reshape(compact_spatial[bdx,:,:,:rank], (-1, rank)),
                    residual[y0:y0 + bheight, x0:x0 + bwidth, :]
                )

        # Decompose Residual
        resid_spatial, _, resid_ranks, resid_indices = batch_decompose(
            fov_height, fov_width, int(t1 - t0), residual,
            params['block_height'], params['block_width'],
            params['spatial_thresh'], params['temporal_thresh'],
            max_remaining_components, params['consec_failures'],
            params['max_iters_main'], params['max_iters_init'],
            params['tol'], params['d_sub'], params['t_sub']
        )

        # Add New Spatial Components To Original Decomp
        for bdx, (prev_rank, resid_rank) in enumerate(zip(block_ranks, resid_ranks)):
            
            # Compute Insert Location
            prev_rank = int(prev_rank)
            resid_rank = int(resid_rank)
            if resid_rank > 0 and prev_rank < params['max_components']:
                
                # Compute New Rank Truncating At Max Components
                new_rank = int(min(prev_rank + resid_rank, params['max_components']))
                slice_size = int(new_rank - prev_rank)
                
                # Append full cumulative basis with new basis vectors
                compact_spatial[bdx, :, :, prev_rank:new_rank] = resid_spatial[bdx, :, :, :slice_size]
                block_ranks[bdx] = new_rank

    return compact_spatial, block_ranks, block_coords


def reformat_spatial(compact_spatial,
                     block_ranks,
                     block_coords,
                     block_weights=None,
                     fov_weights=None,
                     y_offset=0,
                     x_offset=0):
    """ Constructs CSR spatial matrix from compact results of PMD such that
        Y_hat = np.reshape(np.asarray(spatial.dot(temporal)), (fov_height, fov_width, -1))
    """

    # Precompute Dims & Allocate COO Matrix Accordingly
    bheight, bwidth = compact_spatial.shape[1:3]
    nbh, nbw = np.max(block_coords + 1, axis=0)
    fov_height, fov_width = int(nbh * bheight + 2 * y_offset), int(nbw * bwidth + 2 * x_offset)
    total_rank = np.sum(block_ranks)
    sparse_spatial = scipy.sparse.lil_matrix((fov_width * fov_height, total_rank), dtype=np.float32)

    # Default Parameter Setting: Constant Weights & Scales, empty cumulative weights
    if block_weights is None:
        block_weights = np.ones((bheight, bwidth))

    if fov_weights is None:
        fov_weights = np.zeros((fov_height, fov_width))

    ## Accumulate Scaled Blocks
    r0 = 0
    for bdx, rank in enumerate(block_ranks):

        ## Compute Coord Offsets (wrt. FOV) & Rank of Block
        ydx = int(bheight * block_coords[bdx, 0] + y_offset)
        xdx = int(bwidth * block_coords[bdx, 1] + x_offset)
        rank = int(rank)
        if rank > 0:

            ## Copy Each Block Into Unraveled Full FOV
            row_inc = np.reshape((ydx + np.arange(bheight)) * fov_width, (bheight, 1))
            insert_idx = np.tile(np.arange(bwidth) + xdx, (bheight, 1))
            insert_idx = np.reshape(insert_idx + row_inc, (-1,))
            sparse_spatial[insert_idx, r0:r0 + rank] = np.reshape(
                np.multiply(
                    np.reshape(compact_spatial[bdx, :, :, :rank], (bheight, bwidth, rank)),
                    block_weights[:,:,None]
                ),
                (bheight * bwidth, rank)
            )

        ## Increment Cumulative Weight Image & Rank Counter
        fov_weights[ydx:ydx+bheight, xdx: xdx + bwidth] += block_weights
        r0 += rank

    # Return CSR For Efficient Pixelwise Slicing (Used In HemoCorr)
    return sparse_spatial.tocsr(), fov_weights


def fit_spatial(params, data):
    """ """
    if params['overlapping']:

        # Compute Dimensions Of Problem 
        fov_height, fov_width = params['fov_height'], params['fov_width']
        bheight, bwidth = params['block_height'], params['block_width']
        hbh = bheight // 2
        hbw = bwidth // 2
        
        # Construct Pyramidal Weights For Each Block & FOV Accumulator
        block_weights = np.minimum(
            np.tile(np.arange(1, 1 + hbw), (hbh, 1)),
            np.tile(np.arange(1, 1 + hbh), (hbw, 1)).T
        )
        block_weights = np.hstack([block_weights, np.fliplr(block_weights)])
        block_weights = np.vstack([block_weights, np.flipud(block_weights)])
        fov_weights = np.zeros((fov_height, fov_width))

        # Iterate Over 4 Tiling Skews (Standard, Vert, Horz, Diag)
        sparse_spatial_mats = []
        for y0, x0 in [(0,0), (hbh, 0), (0, hbw), (hbh, hbw)]:
            display(' --- Iterating slice ({0},{1})'.format(y0,x0))
            # Decompose Slice Of FOV
            compact_spatial, block_ranks, block_indices = windowed_pmd(
                params, data[y0:fov_height - y0, x0:fov_width - x0, :]
            )

            # Reformat Results Into New Sparse Spatial Matrix
            sparse_spatial, fov_weights = reformat_spatial(
                compact_spatial,
                block_ranks,
                block_indices,
                block_weights=block_weights,
                fov_weights=fov_weights,
                y_offset=y0,
                x_offset=x0
            )
            sparse_spatial_mats.append(sparse_spatial)

        # Append All Spatial Matrices & Scale By Cumulative Weights
        sparse_spatial = scipy.sparse.hstack(sparse_spatial_mats, format="csr", dtype=np.float32) 
        normalizing_weights = scipy.sparse.diags([(1 / fov_weights).astype(np.float32).ravel()], [0])
        sparse_spatial = normalizing_weights.dot(sparse_spatial)
    
    else:
        
        # Single Decomposition Without Shifting
        display('Performing single decomposition without shifting.')
        compact_spatial, block_ranks, block_indices = windowed_pmd(params, data)
        sparse_spatial, _ = reformat_spatial(compact_spatial,
                                             block_ranks,
                                             block_indices)

    # Only Return Value Is Spatial Matrix
    return sparse_spatial


def dense_regression(X, Y):
    """
    X: dense Matrix
    Y: dense matrix
    """
    Sigma = X.T.dot(X)
    Theta = np.linalg.inv(Sigma)
    return Theta.dot(X.T.dot(Y)).astype(np.float32)


def sparse_regression(X, Y):
    """
    X: CSR or CSC Matrix
    Y: dense matrix
    """
    display("running: Sigma = X.T.dot(X)...")
    Sigma = X.T.dot(X)
    display("running: Sigma = scipy.sparse.linalg.inv(Sigma)...")
    Theta = scipy.sparse.linalg.inv(Sigma)
    display("running Theta = theta.todense()")
    Theta = np.asarray(Theta.todense())
    display("running: tmp = X.dot(Theta.T)")
    tmp = X.dot(Theta.T)
    display("{}".format(type(tmp)))
    display("running: return tmp.T.dot(Y.astype(np.float32))...")
    return tmp.T.dot(Y.astype(np.float32))  # FUTURE: convert data before this step


def post_selection_svd(main_spatial,   # sparse
                       main_temporal,  # dense
                       bg_spatial,     # dense
                       bg_temporal):   # dense
    """ Rearrange arbitrary matrix factorization into SVD form
        ... new approach needs new description
    """

    # Step 0: Aggregate Temporal Components
    if bg_temporal is not None:
        temporal = np.vstack([bg_temporal, main_temporal])
        bg_rank = background[1].shape[0]
    else:
        temporal = main_temporal
        bg_rank = 0
    total_rank = temporal.shape[0]

    # Step 1: Othogonalize Temporal Components LQ = V
    qr_out = np.linalg.qr(temporal.T)
    Q = qr_out[0].T
    L = qr_out[1].T

    # Step 2: Fast Transformed Spatial Inner Product Sigma = L'U'UL
    Sigma = np.empty((total_rank, total_rank), dtype=np.float32)
    Sigma[bg_rank:, bg_rank:] = np.asarray(main_spatial.T.dot(main_spatial).todense())
    if bg_rank > 0:
        Sigma[:bg_rank, :bg_rank] = bg_spatial.T.dot(bg_spatial)
        Sigma[bg_rank:, :bg_rank] = main_spatial.T.dot(bg_spatial)
        Sigma[:bg_rank, bg_rank:] = Sigma[bg_rank:, :bg_rank].T
    Sigma = np.dot(Sigma, L)
    Sigma = np.dot(L.T, Sigma)

    # Step 3: Eigen Decomposition Of Sigma
    eig_vals, eig_vecs = np.linalg.eigh(Sigma)
    sing_vals = np.sqrt(eig_vals)

    # Step 5: Construct Orthonormal Bases Explicitly
    mixer = L.dot(eig_vecs)
    mixer /= sing_vals[None, :]
    spatial_basis = main_spatial.dot(mixer[bg_rank:, :])
    if bg_rank > 0:
        spatial_basis += np.dot(bg_spatial, mixer[:bg_rank, :])
    temporal_basis = eig_vecs.T.dot(Q)

    return spatial_basis, sing_vals, temporal_basis  # U, s, Vt


def prune_svd(spatial_basis,
              singular_values,
              temporal_basis,
              variance_threshold,
              ascending=True):
    """ 
    assumed to be in ascending order of singular_values
    """

    # Choose Pruned Rank
    component_variances = np.power(singular_values[-1::-1], 2)
    variance_explained = np.cumsum(component_variances) / np.sum(component_variances)
    pruned_rank = np.argmax(variance_explained > variance_threshold)

    # Format Outputs -- Cut From Smallest Side
    if ascending:
        keep_idx = singular_values.shape[0] - pruned_rank
        return (spatial_basis[:, keep_idx:],
                singular_values[keep_idx:],
                temporal_basis[keep_idx:, :])

    return (spatial_basis[:, :pruned_rank],
            singular_values[:pruned_rank],
            temporal_basis[:pruned_rank, :])


def convert_to_numpy(d):
    for key in d.keys():
        if type(d[key]) is dict:
            convert_to_numpy(d[key])
        else:
            d[key] = np.asarray(d[key])



def save_pmd_results(sparse_spatial, outdir):
    """ Wite PMD results as series of npz files """

    # Sparse Matrix Must Be Saved/Loaded By Scipy Utils
    scipy.sparse.save_npz(os.path.join(outdir, "sparse_spatial.npz"),
                          sparse_spatial,
                          compressed=True)


def display(msg):
    sys.stdout.write('['+datetime.datetime.today().strftime('%y-%m-%d %H:%M:%S')+'] - ' + msg + '\n')
    sys.stdout.flush()


if __name__ == "__main__":
    
    # Avoids Hanging Bug From Wfield Multiprocessing Calls
    #multiprocessing.set_start_method("spawn")

    ## Data Preparation

    # Get Command Line Args
    filename, ext = sys.argv[1].split('.')
    indir = sys.argv[2]
    outdir = sys.argv[3]
    # Read Params & Data
    params = load_params(os.path.join(indir, CONFIG_NAME), CONFIG_EXT)

    # Data Loading
    data = load_data(indir, filename, ext)

    # Validate Params Across Data & Simulate Thresholds If Needed
    display("Validating user-provided parameters...")
    validate_param_compatibility(params, data)
    sim_flag = simulate_missing_params(params)
    display("Writing full parameter set to 'config.yaml'...")
    write_params(os.path.join(outdir, CONFIG_NAME), CONFIG_EXT, params, sim_flag)
    display("Parameter processing completed.")

    # Decomposition Pre-Processing
    # FUTURE: allow for user-specified cropping of a FOV
    # FUTURE: allow for user-specified detrending methods (splines, lowpass, poly regreggion, ect.)
    # FUTURE: allow for float32 data type in preprocessing (noise estimation)

    # Standardization
    display("Performing Standarization...")
    data, baseline, scale = standardize(data,
                                        center=params['center'],
                                        scale=params['scale'])
    display("Standardization completed. Saving intermediate results...")
    np.savez_compressed(os.path.join(outdir, "standardization_images.npz"),
                        mean=baseline,
                        stdv=scale)
    display("Pixelwise mean and estimated noise stdv saved as 'standardization_images.npz'.")

    # Remove Background
    display("Removing Background from standardized data...")
    data, standardized_background = extract_background(params, data)
    display("Background removed. Saving intermediate results...")
    np.save(os.path.join(outdir, 'standardized_spatial_background.npy'),
            standardized_background[0])
    display("Standardized spatial background components saved as 'standardized_spatial_background.npy'.")
    np.save(os.path.join(outdir, 'standardized_temporal_background.npy'),
            standardized_background[1])
    display("Standardized temporal background components saved as 'standardized_temporal_background.npy'.")

    # Incrementally Fit Spatial Components To Rolling Windows Of Time
    display("Performing windowed PMD on standardized data...")
    standardized_sparse_spatial = fit_spatial(params, data)
    display("Windowed PMD completed. Saving intermediate result...")
    scipy.sparse.save_npz(os.path.join(outdir, "standardized_sparse_spatial.npz"),
                          standardized_sparse_spatial,
                          compressed=True)
    display("Standardized spatial matrix saved as 'standardized_sparse_spatial.npz'.")

    # Fit Temporal Basis From Full, Standardized Data
    display("Fitting temporal components to cumulative spatial fit...")
    standardized_temporal = sparse_regression(
        standardized_sparse_spatial, np.reshape(data, (-1, data.shape[-1]))
    )
    display("Regression completed. Saving intermediate result...")
    np.save(os.path.join(outdir, 'standardized_temporal.npy'),
            standardized_temporal)
    display("Standardized temporal components saved as 'standardized_temporal.npy'.")

    # Post-Decomposition Reformatting

    # TODO: Best way/whether to add back the mean image?
    # Rescale & Renormalize Components
    display("Rescaling components ...")
    diag_scale = scipy.sparse.diags([scale.ravel()], [0])
    sparse_spatial = diag_scale.dot(standardized_sparse_spatial)
    if params['background_rank'] > 0:
        background = (standardized_background[0] * scale.ravel()[:, None],
                      standardized_background[1])
    else:
        background = (None, None)

    # SVD of Denoised Movie
    display("Performing SVD reformat of cumulative decomposition...")
    spatial_basis, singular_values, temporal_basis = post_selection_svd(
        sparse_spatial, standardized_temporal, background[0], background[1]
    )
    display("SVD reformat completed. Saving intermediate result...")
    np.savez(os.path.join(indir, "overcomplete_decomposition.npz"),
             U=np.reshape(spatial_basis,
                          (params['fov_height'], params['fov_width'], -1)),
             s=singular_values,
             Vt=temporal_basis)
    display("SVD reformatted results saved as 'overcomplete_decomposition.npz'...")

    # Pruning (Rank Reduction) Of Results
    if params['post_selection_threshold'] < 1:
        display("Pruning overcomplete decomposition...")
        pruned_spatial, pruned_values, pruned_temporal = prune_svd(
            spatial_basis,
            singular_values,
            temporal_basis,
            params['post_selection_threshold']
        )
        display("Prune completed. Saving results...")
        np.savez(os.path.join(indir, "decomposition.npz"),
                 U=np.reshape(pruned_spatial,
                              (params['fov_height'], params['fov_width'], -1)),
                 s=pruned_values,
                 Vt=pruned_temporal)
        display("Pruned decomposition saved as  as 'decomposition.npz'.")

    display('Completed, returning results and shutting down.')

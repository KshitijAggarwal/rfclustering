import numpy as np
from utils.clustering_utils import unison_shuffled_copies, get_data
import os

def save_one(clean_pkl, rfi_pkl=None, rfi_frac=None, downsample=1, outdir = 'dataset/'):
    c_pkl = clean_pkl
    c_data, c_labels, c_snrs = get_data(c_pkl, downsample, frac=1, label=1)

    if rfi_pkl:
        rand_rfi_pkl = rfi_pkl
        if not rfi_frac: 
            rfi_frac = np.random.uniform(0.2, 1)
        r_pkl = rand_rfi_pkl
        r_data, r_labels, r_snrs = get_data(r_pkl, downsample, frac=rfi_frac, label=-1)
    
        data = np.concatenate((r_data, c_data))
        labels = np.concatenate((r_labels, c_labels))
        snrs = np.concatenate((r_snrs, c_snrs))
        name = f'clean_{len(c_data)}_rfi_{len(r_data)}_frac_{rfi_frac:f}.npz'

    else:
        if c_data.any():
            data = c_data
            labels = c_labels
            snrs = c_snrs
            bn = os.path.basename(c_pkl)
            name = f'{bn}_clean_{len(c_data)}.npz'
        else:
            return None

    d, l, s = unison_shuffled_copies(data, labels, snrs)
    
    np.savez(outdir + name, cands=d, labels=l, snrs=s)
    return name


def save_one_wrt_rfi_frac(clean_pkl, rfi_pkl=None, rfi_frac=None, downsample=1, outdir = 'dataset/'):
    c_pkl = clean_pkl
    c_data, c_labels, c_snrs = get_data(c_pkl, downsample, frac=1, label=1)

    rand_rfi_pkl = rfi_pkl
    r_pkl = rand_rfi_pkl
    r_data, r_labels, r_snrs = get_data(r_pkl, downsample, frac=1, label=-1)

    n_frb = c_data.shape[0] 
    n_rfi = r_data.shape[0]
    if not rfi_frac:
        max_rfi_frac = n_rfi / (n_frb + n_rfi)
        rfi_frac = np.random.uniform(0.1, max_rfi_frac)
    size = rfi_frac*n_frb/(1-rfi_frac)
    if size > n_rfi:
        return None
        
    indx = np.random.choice(r_data.shape[0], size=int(size), replace=False)
    r_data_use = np.take(r_data, indx, axis=0)    
    r_snrs_use = np.take(r_snrs, indx, axis=0)
    r_labels_use = np.take(r_labels, indx, axis=0)
    
    data = np.concatenate((r_data_use, c_data))
    labels = np.concatenate((r_labels_use, c_labels))
    snrs = np.concatenate((r_snrs_use, c_snrs))
    
    if len(snrs) < 10:
        return None
    
    name = f'clean_{len(c_data)}_rfi_{len(r_labels_use)}_frac_{rfi_frac:f}.npz'

    d, l, s = unison_shuffled_copies(data, labels, snrs)
    
    np.savez(outdir + name, cands=d, labels=l, snrs=s)
    return name, clean_pkl, rfi_pkl, rfi_frac
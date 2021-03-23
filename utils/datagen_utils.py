import os

import numpy as np

from utils.clustering_utils import unison_shuffled_copies, get_data


def save_one(clean_pkl, rfi_pkl=None, rfi_frac=None, downsample=1, outdir="dataset/"):
    """
    Generates one example for clustering.

    Read two pickles, one with RFI and other with FRB candidates, preprocesses for clustering,
    selects a fraction of RFI candidates and concatenates the two datasets.

    :param clean_pkl: Pickle with only FRB candidates
    :param rfi_pkl: Pickle with only RFI candidates
    :param rfi_frac: Fraction of RFI candidates to use
    :param downsample: Downsampling factor for image features
    :param outdir: Output directory
    :return:
    """
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
        name = f"clean_{len(c_data)}_rfi_{len(r_data)}_frac_{rfi_frac:f}.npz"

    else:
        if c_data.any():
            data = c_data
            labels = c_labels
            snrs = c_snrs
            bn = os.path.basename(c_pkl)
            name = f"{bn}_clean_{len(c_data)}.npz"
        else:
            return None

    d, l, s = unison_shuffled_copies(data, labels, snrs)

    np.savez(outdir + name, cands=d, labels=l, snrs=s)
    return name


def save_one_wrt_rfi_frac(
    clean_pkl, rfi_pkl=None, rfi_frac=None, downsample=1, outdir="dataset/"
):
    """
    Same as the previous function, but the rfi_fraction is wtih respect to the total number of candidates.

    Generates one example for clustering.
    Read two pickles, one with RFI and other with FRB candidates, preprocesses for clustering,
    selects a fraction of RFI candidates and concatenates the two datasets.

    :param clean_pkl: Pickle with only FRB candidates
    :param rfi_pkl: Pickle with only RFI candidates
    :param rfi_frac: Fraction of RFI candidates to use
    :param downsample: Downsampling factor for image features
    :param outdir: Output directory
    :return:
    """
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
    size = rfi_frac * n_frb / (1 - rfi_frac)
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

    name = f"clean_{len(c_data)}_rfi_{len(r_labels_use)}_frac_{rfi_frac:f}.npz"

    d, l, s = unison_shuffled_copies(data, labels, snrs)

    np.savez(outdir + name, cands=d, labels=l, snrs=s)
    return name, clean_pkl, rfi_pkl, rfi_frac

#!/usr/bin/env python
# coding: utf-8

import argparse
import glob
import json

import numpy as np
import tqdm
from sklearn import cluster

import hs_utils
from clustering_utils import calculate_metric_terms


def model(**params):
    """
    Run clustering on all the observations and calculate metrics and final score.

    :param params: Input parameters of clustering function
    :return:
    """
    N_frb = 0
    vs = []
    hs = []
    cs = []
    ncands = []

    for cand in candlist:
        c_res = calculate_metric_terms(
            cand, cluster_function=cluster_function, debug=False, plot=False, **params
        )
        t, frb_found, h, c, v = c_res
        vs.append(v)
        hs.append(h)
        cs.append(c)
        ncands.append(t)

        if frb_found:
            N_frb += 1

    vs = np.array(vs)
    hs = np.array(hs)
    cs = np.array(cs)
    c_avg = np.average(cs, axis=0, weights=ncands)
    h_avg = np.average(hs, axis=0, weights=ncands)
    v_avg = np.average(vs, axis=0, weights=ncands)
    recall = N_frb / len(vs)
    score = v_avg * recall

    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run random sampling on the dataset using a specified method",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        help="Clustering algorithm to use (hdbscan, dbscan, kmeans, affinity_prop, "
        "meanshift, meanshift_norm, agglo, sc, optics, birch)",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-o",
        "--outdir",
        help="Output directory",
        required=False,
        type=str,
        default="/hyrule/data/users/kshitij/hdbscan/rfclustering/hs_results/",
    )
    parser.add_argument(
        "-d", "--data", help="path of test data", required=True, type=str
    )
    parser.add_argument(
        "-e", "--ext", help="Extension to the outname", required=True, type=str
    )
    values = parser.parse_args()

    print("Input Arguments:-")
    for arg, value in sorted(vars(values).items()):
        print(f"Argument {arg}: {value}")

    global cluster_function

    if values.algorithm == "hdbscan":
        from hdbscan import HDBSCAN

        list_of_samples = hs_utils.search_space_hdbscan()
        cluster_function = HDBSCAN
    elif values.algorithm == "dbscan":
        list_of_samples = hs_utils.search_space_dbscan()
        cluster_function = cluster.DBSCAN
    elif values.algorithm == "kmeans":
        list_of_samples = hs_utils.search_space_kmeans()
        cluster_function = cluster.KMeans
    elif values.algorithm == "affinity_prop":
        list_of_samples = hs_utils.search_space_ap()
        cluster_function = cluster.AffinityPropagation
    elif values.algorithm == "meanshift":
        list_of_samples = hs_utils.search_space_ms()
        cluster_function = cluster.MeanShift
    elif values.algorithm == "meanshift_norm":
        list_of_samples = hs_utils.search_space_ms_norm()
        cluster_function = cluster.MeanShift
    elif values.algorithm == "agglo":
        list_of_samples = hs_utils.search_space_agglo()
        cluster_function = cluster.AgglomerativeClustering
    elif values.algorithm == "sc":
        list_of_samples = hs_utils.search_space_sc()
        cluster_function = cluster.SpectralClustering
    elif values.algorithm == "optics":
        list_of_samples = hs_utils.search_space_optics()
        cluster_function = cluster.OPTICS
    elif values.algorithm == "birch":
        list_of_samples = hs_utils.search_space_birch()
        cluster_function = cluster.Birch
    else:
        print("Can't run this algorithm")

    files = glob.glob(f"{values.data}/*npz")
    candlist = []
    for file in files:
        f = np.load(file)
        cand = {}
        cand["cands"] = f["cands"]
        cand["labels"] = f["labels"]
        cand["snrs"] = f["snrs"]
        candlist.append(cand)

    outlist = []
    for i, d in tqdm.tqdm(enumerate(list_of_samples)):
        s = model(**d)
        dict_with_score = d
        dict_with_score["score"] = s
        outlist.append(dict_with_score)
        if not i % 100:
            with open(
                f"{values.outdir}/test_{values.algorithm}_{values.ext}.json", "w"
            ) as fout:
                json.dump(outlist, fout, indent=4)

    with open(
        f"{values.outdir}/{values.algorithm}_{values.ext}_result_grid.json", "w"
    ) as fout:
        json.dump(outlist, fout, indent=4)

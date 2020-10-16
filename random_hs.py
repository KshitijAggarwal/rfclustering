#!/usr/bin/env python
# coding: utf-8

import json
import glob
import argparse 

from sklearn import cluster

import hs_utils
from clustering_utils import calculate_metric_terms

import tqdm
import numpy as np
        
def model(**params):
    N_frb = 0
    vs = []
    hs = []
    cs = []
    ncands = []

    for cand in candlist:
        c_res = calculate_metric_terms(cand, cluster_function=cluster_function, debug=False, 
                                       plot=False, **params)
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
    recall = N_frb/len(files)
    score = v_avg*recall

    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run random sampling on the dataset using a specified method",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-a', '--algorithm', help='Clustering algorithm to use (hdbscan, dbscan, kmeans, affinity_prop, '
                       'meanshift, agglo)', required=True, type=str)
    parser.add_argument('-o', '--outdir', help='Output directory', required=False, type=str, 
                        default='/hyrule/data/users/kshitij/hdbscan/rfclustering/hs_results/')
    values = parser.parse_args()  
    
    print("Input Arguments:-")
    for arg, value in sorted(vars(values).items()):
        print(f"Argument {arg}: {value}")
            
    global cluster_function

    if values.algorithm == 'hdbscan':
        from hdbscan import HDBSCAN
        list_of_samples = hs_utils.search_space_hdbscan()
        cluster_function = HDBSCAN
    elif values.algorithm == 'dbscan':
        list_of_samples = hs_utils.search_space_dbscan()
        cluster_function = cluster.DBSCAN
    elif values.algorithm == 'kmeans':
        list_of_samples = hs_utils.search_space_kmeans()
        cluster_function = cluster.KMeans
    elif values.algorithm == 'affinity_prop':
        list_of_samples = hs_utils.search_space_ap()
        cluster_function = cluster.AffinityPropagation
    elif values.algorithm == 'meanshift':
        list_of_samples = hs_utils.search_space_ms()
        cluster_function = cluster.MeanShift        
    elif values.algorithm == 'agglo':
        list_of_samples = hs_utils.search_space_agglo()
        cluster_function = cluster.AgglomerativeClustering        
    else:
        print("Can't run this algorithm")
        
    files = glob.glob('/hyrule/data/users/kshitij/hdbscan/rfclustering/test_data_200/*npz')
    candlist = []
    for file in files:
        f = np.load(file)
        cand = {}
        cand['cands'] = f['cands']
        cand['labels'] = f['labels']
        cand['snrs'] = f['snrs']
        #if len(cand['cands']) > 10:
        candlist.append(cand)

    outlist = []
    for i, d in tqdm.tqdm(enumerate(list_of_samples)):
        s = model(**d)
        dict_with_score = d
        dict_with_score['score'] = s
        outlist.append(dict_with_score)
        if not i%100:
            with open(f'{values.outdir}/test_{values.algorithm}.json', 'w') as fout:
                json.dump(outlist, fout, indent=4)

    with open(f'{values.outdir}/{values.algorithm}_result_grid.json', 'w') as fout:
        json.dump(outlist, fout, indent=4)

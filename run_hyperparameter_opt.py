from skopt import Optimizer
from skopt.space import Integer
from skopt.space import Real
from skopt.space import Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence

from hdbscan import HDBSCAN
from sklearn import metrics, cluster
from clustering_utils import calculate_metric_terms
import tqdm, argparse, glob
import numpy as np
import pylab as plt
from joblib import Parallel, delayed
import sys
sys.path.append('/hyrule/data/users/kshitij/hdbscan/rfclustering')
from hyperparam_list import *

# https://machinelearningmastery.com/scikit-optimize-for-hyperparameter-tuning-in-machine-learning/
# https://automl.github.io/auto-sklearn/master/index.html
# https://scikit-optimize.github.io/stable/

def get_args():
    parser = argparse.ArgumentParser(description="Run hyperparameter tuning on the dataset using a specified method",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-a', '--algorithm', help='Clustering algorithm to use (hdbscan, dbscan, kmeans, affinity_prop)', 
                        required=True, type=str)
    parser.add_argument('-o', '--outdir', help='Output directory', required=False, type=str, 
                        default='/hyrule/data/users/kshitij/hdbscan/scripts/final/')
    parser.add_argument('-e', '--estimator', help='Estimator for skopt optimiser', required=False, type=str, default='gp')
    parser.add_argument('-n', '--ncores', help='Number of cores to use', required=False, type=int, default=20)
    parser.add_argument('-l', '--nloops', help='Number of loops to run (total = ncores X nloops)', required=False, 
                        type=int, default=20)

    values = parser.parse_args()  
    
    print("Input Arguments:-")
    for arg, value in sorted(vars(values).items()):
        print(f"Argument {arg}: {value}")

    return values


if __name__ == '__main__':
    values = get_args()

    global cluster_function

    if values.algorithm == 'hdbscan':
        search_space = search_space_hdbscan
        cluster_function = HDBSCAN
    elif values.algorithm == 'dbscan':
        search_space = search_space_dbscan
        cluster_function = cluster.DBSCAN
    elif values.algorithm == 'kmeans':
        search_space = search_space_kmeans
        cluster_function = cluster.KMeans
    elif values.algorithm == 'affinity_prop':
        search_space = search_space_AP
        cluster_function = cluster.AffinityPropagation
    else:
        print("Can't run this algorithm")
    
    @use_named_args(search_space)
    def model(**params):
        files = glob.glob('/hyrule/data/users/kshitij/hdbscan/scripts/final/dataset/*npz')
        
        if values.algorithm == 'kmeans':
            files = [f for f in files if int(f.split('_')[1]) + int(f.split('_')[3]) >= 10]
        
        N_frb = 0
        vs = []
        hs = []
        cs = []
        ncands = []

        for f in files:                
            c_res = calculate_metric_terms(f, cluster_function=cluster_function, debug=False, 
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

        return 1 - score    
        
    optimizer = Optimizer(
        dimensions=search_space,
        random_state=1,
        base_estimator=values.estimator)

    for i in tqdm.tqdm(range(values.nloops)):
        x = optimizer.ask(n_points = values.ncores)  # x is a list of n_points points
        y = Parallel(n_jobs=values.ncores)(delayed(model)(v) for v in x)  # evaluate points in parallel
        optimizer.tell(x, y)

    res = optimizer.get_result()
    print("Best score=%.4f" % res.fun)
    print(res.x)
    plot_convergence(res)
    plt.savefig(f'{values.outdir}/{values.algorithm}_skopt.png', bbox_inches='tight')
    
    from skopt import dump
    dump(res, f'{values.outdir}/{values.algorithm}_skopt.pkl')

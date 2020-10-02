from skopt import Optimizer
from skopt.space import Integer
from skopt.space import Real
from skopt.space import Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence

from hdbscan import HDBSCAN
from sklearn import metrics
from clustering_utils import calculate_metric_terms
import tqdm, argparse, glob
import numpy as np
import pylab as plt
from joblib import Parallel, delayed

search_space = list()
search_space.append(Integer(2, 10, name='min_cluster_size', dtype=int))
search_space.append(Integer(2, 5, name='min_samples',dtype=int))
search_space.append(Categorical(['hamming', 'euclidean', 'canberra', 'chebyshev'], name='metric'))
search_space.append(Categorical(['eom', 'leaf'], name='cluster_selection_method'))
search_space.append(Categorical([True, False], name='allow_single_cluster'))
# search_space.append(Real(0.1, 2, name='cluster_selection_epsilon'))

@use_named_args(search_space)
def model(**params):
    files = glob.glob('dataset/*npz')
    N_frb = 0
    n_cs = []
    n_frb_cs = []
    amis = []
    arands = []
    vs = []
    fms = []
    ps = []
    
    for f in files:                
        c_res = calculate_metric_terms(f, debug=False, plot=False, cluster_function=HDBSCAN, **params)

        n_c, n_frb_c, frb_found, arand, ami, v, fm, p = c_res
        n_cs.append(n_c)
        n_frb_cs.append(n_frb_c)
        amis.append(ami)
        arands.append(arand)
        vs.append(v)
        fms.append(fm)
        ps.append(p)

        if frb_found:
            N_frb += 1

    amis = np.array(amis)
    arands = np.array(arands)
    vs = np.array(vs)
    fms = np.array(fms)
    ps = np.array(ps)
    recall = N_frb/len(files)

    score = np.average(arands)*np.average(ps)*recall
    return 1 - score


n = 20

optimizer = Optimizer(
    dimensions=search_space,
    random_state=1,
    base_estimator='gp',
)

for i in tqdm.tqdm(range(100)):
    x = optimizer.ask(n_points=n)  # x is a list of n_points points
    y = Parallel(n_jobs=n)(delayed(model)(v) for v in x)  # evaluate points in parallel
    optimizer.tell(x, y)

res = optimizer.get_result()
print("Best score=%.4f" % res.fun)
print(res.x)
plot_convergence(res)
plt.savefig('hdbscan_skopt.png')

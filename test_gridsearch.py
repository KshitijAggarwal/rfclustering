#!/usr/bin/env python
# coding: utf-8

# In[1]:


# from sklearn.model_selection import GridSearchCV


# In[20]:


import sys
sys.path.append('../../rfclustering/')
import json


# In[3]:

import glob
from hdbscan import HDBSCAN
from sklearn import metrics
from clustering_utils import calculate_metric_terms
import tqdm
from sklearn import cluster
import numpy as np
import pylab as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:

cluster_function = cluster.SpectralClustering #cluster.AgglomerativeClustering 

#cluster.MeanShift#cluster.AffinityPropagation #cluster.DBSCAN #cluster.KMeans
# cluster_function = HDBSCAN


# In[6]:


files = glob.glob('/hyrule/data/users/kshitij/hdbscan/scripts/final/dataset/*npz')
candlist = []
for file in files:
    f = np.load(file)
    cand = {}
    cand['cands'] = f['cands']
    cand['labels'] = f['labels']
    cand['snrs'] = f['snrs']
    if len(cand['cands']) > 10:
        candlist.append(cand)


# In[7]:

def random_sample(l, n):
    assert len(l) > 1
    assert n
    if l[0] == 'int':
        if len(l) == 3:
            return np.random.randint(l[1], l[2]+1, n, dtype='int32')
        elif len(l) == 2:
            return np.random.randint(l[1], l[1]+1, n, dtype='int32')
        else:
            return np.random.uniform(l[1], l[2], n)
    elif l[0] == 'cat':
        return np.random.choice(l[1:], size=n)
    elif l[0] == 'float':
        if len(l) == 2:
            return np.random.uniform(l[1], l[1], n)
        else:
            return np.random.uniform(l[1], l[2], n)
    elif l[0] == 'loguniform_int':
        if len(l) == 2:
            return np.power(10, (np.random.randint(np.log10(l[1]), np.log10(l[1]), n)), dtype=float)
        else:
            return np.power(10, (np.random.randint(np.log10(l[1]), np.log10(l[2]) + 1, n)), dtype=float)
    elif l[0] == 'loguniform':
        if len(l) == 2:
            return 10**(np.random.uniform(np.log10(l[1]), np.log10(l[1]), n))            
        else:
            return 10**(np.random.uniform(np.log10(l[1]), np.log10(l[2]) + 1), n)
    else:
        raise ValueError('Something went wrong')
        
def model(**params):

#     if values.algorithm == 'kmeans':
#     files = [f for f in files if int(f.split('_')[1]) + int(f.split('_')[3]) >= 10]

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

    return 1 - score


# In[71]:

# search_space_kmeans = []
# param_d = {}
# param_d['n_clusters'] = ['int', 2, 10]
# param_d['n_init'] = ['int', 10, 30]
# param_d['algorithm'] = ['cat', 'auto', 'full', 'elkan']
# param_d['random_state'] = ['int', 1996]
# search_space_kmeans.append(param_d)

# search_space_hdbscan = []
# param_d = {}
# param_d['min_cluster_size'] = ['int', 2, 10]
# param_d['min_samples'] = ['int', 2, 5]
# param_d['metric'] = ['cat', 'euclidean', 'hamming', 'chebyshev', 'cityblock', 'manhattan', 'canberra']
# param_d['cluster_selection_method'] = ['cat', 'eom', 'leaf']
# param_d['allow_single_cluster'] = ['cat', True, False]
# search_space_hdbscan.append(param_d)


# In[72]:
# search_space_dbscan = []
# param_d = {}
# param_d['min_samples'] = ['int', 2, 10]
# param_d['eps'] = ['float', 0.5, 10]
# param_d['metric'] = ['cat', 'euclidean', 'chebyshev', 'cityblock', 'manhattan', 'canberra']
# param_d['algorithm'] = ['cat', 'auto']
# param_d['leaf_size'] = ['int', 20, 40]
# search_space_dbscan.append(param_d)


# n = 6000
# samples = {}
# for p in param_d.keys():
#     s = random_sample(param_d[p], n)
#     samples[p] = s.tolist()

# list_of_samples_others = [dict(zip(samples,t)) for t in zip(*samples.values())]


# param_d['eps'] = ['float', 0.01, 1]
# param_d['metric'] = ['cat', 'hamming']

# n = 2000
# samples = {}
# for p in param_d.keys():
#     s = random_sample(param_d[p], n)
#     samples[p] = s.tolist()

# list_of_samples_hamming = [dict(zip(samples,t)) for t in zip(*samples.values())]

# list_of_samples = list_of_samples_hamming + list_of_samples_others


# search_space_AP = []
# param_d = {}
# param_d['damping'] = ['float', 0.5, 1]
# param_d['affinity'] = ['cat', 'euclidean']
# param_d['preference'] = ['int', -1000, -200]
# search_space_AP.append(param_d)

# search_space_MS = []
# param_d = {}
# param_d['bandwidth'] = ['float', 10, 40]
# param_d['bin_seeding'] = ['cat', True, False]
# param_d['cluster_all'] = ['cat', True, False]
# search_space_MS.append(param_d)

# search_space_agglo = []
# param_d = {}
# param_d['n_clusters'] = ['int', 2, 10]
# param_d['affinity'] = ['cat', 'euclidean', 'l1', 'l2', 
#                        'manhattan', 'cosine']
# param_d['compute_full_tree'] = ['cat', 'auto']
# param_d['linkage'] = ['cat', 'complete', 'average', 'single']
# search_space_agglo.append(param_d)

# n = 400

# samples = {}
# for p in param_d.keys():
#     s = random_sample(param_d[p], n)
#     samples[p] = s.tolist()

# list_of_samples_others = [dict(zip(samples,t)) for t in zip(*samples.values())]

# n = 100
# param_d['affinity'] = ['cat', 'euclidean']
# param_d['linkage'] = ['cat', 'ward']

# # search space: 8*5*1*4 = 160 (split into two, as ward only takes euclidean)

# samples = {}
# for p in param_d.keys():
#     s = random_sample(param_d[p], n)
#     samples[p] = s.tolist()
# list_of_samples_ward = [dict(zip(samples,t)) for t in zip(*samples.values())]
# list_of_samples = list_of_samples_others + list_of_samples_ward

search_space_SC = []
param_d = {}
param_d['n_clusters'] = ['int', 2, 10]
# param_d['n_components'] = ['int', 2, 10]
param_d['eigen_solver'] = ['cat', 'arpack', 'lobpcg']
param_d['random_state'] = ['int', 1996]
param_d['gamma'] = ['loguniform_int', 10**(-7), 10**(-2)]
param_d['affinity'] = ['cat', 'nearest_neighbors', 'rbf', 'additive_chi2',
                       'chi2', 'linear', 'poly', 'laplacian', 'sigmoid', 
                       'cosine']
param_d['assign_labels'] = ['cat', 'kmeans', 'discretize']
search_space_SC.append(param_d)

# # search space: 8*8*2*1*6*8*2 = 12288
# # search space: 8*2*1*6*9*2 = 1728

n = 2000

samples = {}
for p in param_d.keys():
    s = random_sample(param_d[p], n)
    samples[p] = s.tolist()

list_of_samples = [dict(zip(samples,t)) for t in zip(*samples.values())]

# In[26]:

outlist = []
for i, d in tqdm.tqdm(enumerate(list_of_samples)):
    s = model(**d)
    dict_with_score = d
    dict_with_score['score'] = s
    outlist.append(dict_with_score)
#     list_of_samples[i]['score'] = s
    if not i%100:
        with open('test_SC.json', 'w') as fout:
            json.dump(outlist, fout, indent=4)

# In[28]:

with open('SC_result_grid.json', 'w') as fout:
    json.dump(outlist, fout, indent=4)


# In[83]:


#v = {k: [dic[k] for dic in list_of_samples] for k in list_of_samples[0]}


# In[39]:


#plt.scatter(v['n_clusters'], v['score'])


# In[31]:


# points_n_clusters = np.random.randint(param_d['n_clusters'][0], param_d['n_clusters'][1], n)
# points_n_init = np.random.randint(param_d['n_init'][0], param_d['n_init'][1], n)
# points_algorithm = np.random.choice(param_d['algorithm'], size=1)[0]

# search_space_kmeans = list()
# #add random seed 
# search_space_kmeans.append(Integer(2, 10, name='n_clusters', dtype=int))
# # search_space_kmeans.append(Categorical(['k-means++', 'random'], name='init'))
# search_space_kmeans.append(Integer(10, 30, name='n_init', dtype=int))
# search_space_kmeans.append(Categorical(['auto', 'full', 'elkan'], name='algorithm'))
# search_space_kmeans.append(Real(1e-5, 1e-3, 'log-uniform',  name='tol'))
# search_space_kmeans.append(Categorical([1996], name='random_state'))
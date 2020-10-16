import numpy as np

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

        
def get_list_of_samples(param_dict, n):
    samples = {}
    for p in param_dict.keys():
        s = random_sample(param_dict[p], n)
        samples[p] = s.tolist()

    list_of_samples = [dict(zip(samples,t)) for t in zip(*samples.values())]
    return list_of_samples

        
def search_space_kmeans():
    n = 1000
    param_d = {}
    param_d['n_clusters'] = ['int', 2, 10]
    param_d['n_init'] = ['int', 10, 30]
    param_d['algorithm'] = ['cat', 'auto', 'full', 'elkan']
    param_d['random_state'] = ['int', 1996]
    return get_list_of_samples(param_d, n)


def search_space_hdbscan():
    n = 1000
    param_d = {}
    param_d['min_cluster_size'] = ['int', 2, 10]
    param_d['min_samples'] = ['int', 2, 5]
    param_d['metric'] = ['cat', 'euclidean', 'hamming', 'chebyshev', 
                         'cityblock', 'manhattan', 'canberra']
    param_d['cluster_selection_method'] = ['cat', 'eom', 'leaf']
    param_d['allow_single_cluster'] = ['cat', True, False]
    return get_list_of_samples(param_d, n)


def search_space_dbscan():
    n = 6000
    param_d = {}
    param_d['min_samples'] = ['int', 2, 10]
    param_d['eps'] = ['float', 0.5, 10]
    param_d['metric'] = ['cat', 'euclidean', 'chebyshev', 'cityblock', 'manhattan', 
                         'canberra']
    param_d['algorithm'] = ['cat', 'auto']
    param_d['leaf_size'] = ['int', 20, 40]
    ls1 = get_list_of_samples(param_d, n)
    
    n = 2000
    param_d['eps'] = ['float', 0.01, 1]
    param_d['metric'] = ['cat', 'hamming']
    ls2 = get_list_of_samples(param_d, n)
    
    return ls1 + ls2


def search_space_ap():
    n = 1000
    param_d = {}
    param_d['damping'] = ['float', 0.5, 1]
    param_d['affinity'] = ['cat', 'euclidean']
    param_d['preference'] = ['int', -1000, -200]
    return get_list_of_samples(param_d, n)


def search_space_ms():
    n = 500 
    param_d = {}
    param_d['bandwidth'] = ['float', 10, 40]
    param_d['bin_seeding'] = ['cat', True, False]
    param_d['cluster_all'] = ['cat', True, False]
    return get_list_of_samples(param_d, n)


def search_space_agglo():
    n = 400 
    param_d ={}
    param_d['n_clusters'] = ['int', 2, 10]
    param_d['affinity'] = ['cat', 'euclidean', 'l1', 'l2', 
                           'manhattan', 'cosine']
    param_d['compute_full_tree'] = ['cat', 'auto']
    param_d['linkage'] = ['cat', 'complete', 'average', 'single']
    ls1 = get_list_of_samples(param_d, n)

    n = 100
    param_d['affinity'] = ['cat', 'euclidean']
    param_d['linkage'] = ['cat', 'ward']
    ls2 = get_list_of_samples(param_d, n)

    return ls1 + ls2    

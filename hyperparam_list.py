from skopt.space import Integer
from skopt.space import Real
from skopt.space import Categorical

search_space_hdbscan = list()
search_space_hdbscan.append(Integer(2, 10, name='min_cluster_size', dtype=int))
search_space_hdbscan.append(Integer(2, 5, name='min_samples',dtype=int))
search_space_hdbscan.append(Categorical(['euclidean', 'hamming', 'chebyshev', 'cityblock', 
                                        'manhattan', 'canberra'], name='metric'))
search_space_hdbscan.append(Categorical(['eom', 'leaf'], name='cluster_selection_method'))
search_space_hdbscan.append(Categorical([True, False], name='allow_single_cluster'))
# search_space.append(Real(0.2, 4, name='cluster_selection_epsilon'))


search_space_kmeans = list()
#add random seed 
search_space_kmeans.append(Integer(2, 10, name='n_clusters', dtype=int))
# search_space_kmeans.append(Categorical(['k-means++', 'random'], name='init'))
search_space_kmeans.append(Integer(10, 30, name='n_init', dtype=int))
search_space_kmeans.append(Categorical(['auto', 'full', 'elkan'], name='algorithm'))
search_space_kmeans.append(Real(1e-5, 1e-3, 'log-uniform',  name='tol'))
search_space_kmeans.append(Categorical([1996], name='random_state'))


search_space_AP = list()
# maybe play with preference value a little bit? can even be negative, very sensitive to it
search_space_AP.append(Real(0.5, 1, name='damping'))
search_space_AP.append(Categorical(['euclidean’, ‘precomputed'], name='affinity'))
search_space_AP.append(Integer(10, 20, name='convergence_iter', dtype=int))
search_space_AP.append(Categorical('int', name='random_state'))


search_space_dbscan = list()
search_space_dbscan.append(Real(0.1, 4, name='eps'))
search_space_dbscan.append(Integer(2, 10, name='min_samples', dtype=int))
search_space_dbscan.append(Categorical(['euclidean', 'hamming', 'chebyshev', 'cityblock', 
                                        'manhattan', 'canberra'], name='metric'))
search_space_dbscan.append(Categorical(['auto'], name='algorithm'))
search_space_dbscan.append(Integer(10, 50, name='leaf_size', dtype=int))
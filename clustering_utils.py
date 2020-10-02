from copy import deepcopy
import numpy as np
from rfpipe import candidates
import glob, logging, os
from sklearn import metrics
from cluster_plot import plot_data

def calculate_metric_terms(file, cluster_function=None, plot=False, debug=False, **kwargs):
    f = np.load(file)
    d = f['cands']
    l = f['labels']
    s = f['snrs']
    data = d

    assert cluster_function

    # cluster 
    clusterer = cluster_function(**kwargs).fit(data)
    tl = l
    cl = clusterer.labels_

    arand = metrics.adjusted_rand_score(tl, cl)
#     ami = metrics.adjusted_mutual_info_score(tl, cl)
#     v = metrics.v_measure_score(tl, cl)
#     fm = metrics.fowlkes_mallows_score(tl, cl)
    h = metrics.homogeneity_score(tl, cl)

    # for each cluster containing an FRB candidate, calculate its precision
    # defined as number of FRB candidates in that cluster/total number of candidates in that cluster
    # this is to favor clusters which just have FRB and less RFI
    # excluding unclustered candidates
    # if FRB is just unclustered, then precision = 1? 
    
    cluster_labels_frb = list(set(cl[tl == 1]))
#     a = []
    precision = []
    if len(cluster_labels_frb) == 1 and cluster_labels_frb[0] == -1:
            precision.append(0)
    else:
        for c in cluster_labels_frb:
            if c == -1:
                continue
            indexes = np.where(cl == c)[0]
            base_labels = np.take(tl, indexes)
            precision.append((base_labels == 1).sum()/len(base_labels))
#             a.append(indexes)        

    p = np.prod(precision, axis=0)

    if debug:
        print(f'Precisions are {precision}, and clusters with FRBs are:{cluster_labels_frb}')    
        print(file)
        print(f'Adjusted rand score are: {arand}')
#         print(f'Adjusted MI score are: {ami}')
        print(f'Homogenity score is: {h}')
#         print(f'V measures are: {v}')
#         print(f'FM score are: {fm}')
        print(f'Precision of FRB clusters is: {p}')

    if plot:
        plot_data(data, cl, s)
    
    true_labels = tl
    cluster_labels = cl

    N_clusters = np.max(clusterer.labels_ + 1) # excluding unclustered candidates
    N_unclustered = (cluster_labels == -1).sum()
        
    # check if the FRB candidate is recovered or not (after taking the max snr element of each cluster)
    clusters = cluster_labels
    cl_rank = np.zeros(len(clusters), dtype=int)
    cl_count = np.zeros(len(clusters), dtype=int)

    for cluster in np.unique(clusters):
        clusterinds = np.where(clusters == cluster)[0]
        snrs = s[clusterinds]
        cl_rank[clusterinds] = np.argsort(np.argsort(snrs)[::-1]) + 1
        cl_count[clusterinds] = len(clusterinds)

    # rank one is the highest snr candidate of the cluster
    calcinds = np.unique(np.where(cl_rank == 1)[0])

    # i think if this contains 1 i.e FRB candidate, that means that FRB wasn't missed,
    # and some FRB plots will be generated
    
    # Take indexes of rank 1 candididates (from calcinds) in true_labels, and see if 
    # label 1 is in there. If yes, then FRB candidate will pass through in one
    # or more clusters
    if (np.take(true_labels, calcinds) == 1).sum() > 0:
        frb_found = True
    else:
        frb_found = False
    
    actual_frb_indx = np.where(true_labels == 1)[0]
    frb_cand_clustering_labels = np.take(cluster_labels, actual_frb_indx)
    
    # clustering
    N_inspect = N_clusters + N_unclustered

    # pipeline (N_tot, N_frb_cands, N_frb_cluster > 0)
    N_tot = len(true_labels)
    N_frb_cands = (true_labels == 1).sum()
    # number of clusters all the FRB candidates are a part of. This doesn't mean that those clusters will 
    # have an FRB candidate as the max snr candidate. 
    # Therefore, N_frb_cluster may not be equal to (np.take(true_labels, calcinds) == 1).sum()
    N_frb_cluster = len(np.unique(frb_cand_clustering_labels[frb_cand_clustering_labels > -1
                                                            ])) + (frb_cand_clustering_labels == -1).sum()    
    
    # total (0 < n_c < 1, 0 < n_frb_c < 1)
    n_c = N_inspect/N_tot
    n_frb_c = N_frb_cluster/N_frb_cands    
    
    # thresholding the values to 0.9, so that 1 - n_c or 1 - n_frb_c doesn't shrink to zero.
    if n_frb_c > 0.9: 
        n_frb_c = 0.9
        
    if n_c > 0.9: 
        n_frb_c = 0.9
    
    return n_c, n_frb_c, frb_found, arand, h, p


def get_data(pkl, frac=1, label=1):
    cc = list(candidates.iter_cands(pkl, 'candcollection'))[0]
    logging.info(f'Processing {os.path.split(pkl)[0]}')
    _data = prep_to_cluster(cc)
    _snrs = cc.array['snr1']

    size = frac*_data.shape[0]
    if size == 0:
        frac = 1
        size = frac*_data.shape[0]
    indx = np.random.choice(_data.shape[0], size=int(size), replace=False)
#     np.random.randint(0, high=_data.shape[0], size=int(size))
    data = np.take(_data, indx, axis=0)    
    snrs = np.take(_snrs, indx, axis=0)
    labels = label*np.ones(data.shape[0], dtype='int')
    return data, labels, snrs


def unison_shuffled_copies(a, b, c):
    assert len(a) == len(b) == len(c)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]


def find_friends(cc, data, processed_mock, th = 2):
    l_mask = (data[:,0] < processed_mock[0] + th) & (data[:,0] > processed_mock[0] - th)
    m_mask = (data[:,1] < processed_mock[1] + th) & (data[:,1] > processed_mock[1] - th)
    
    locs = np.array(cc.locs)
    friends = locs[l_mask & m_mask]
    others = locs[~(l_mask & m_mask)]

    scanId = cc.state.metadata.scanId
    friend_names = []
    for friend in friends:
        segment, integration, dmind, dtind, beamnum = friend
        friend_names.append('cands_{0}_seg{1}-i{2}-dm{3}-dt{4}.png'.format(scanId, segment, integration, dmind, dtind))
        
    other_names = []    
    for other in others:
        segment, integration, dmind, dtind, beamnum = other
        other_names.append('cands_{0}_seg{1}-i{2}-dm{3}-dt{4}.png'.format(scanId, segment, integration, dmind, dtind))
        
    if len(cc) == len(friend_names):
        logging.info(f'All {len(friend_names)} candidates from simulated transient, no RFI candidate.')
    else:
        logging.info(f'Found {len(friend_names)} simulated candidates out of total {len(cc)} candidates.')
    return friend_names, other_names


def get_mocks(cc, downsample = 2):
    mocks = cc.prefs.simulated_transient
    st = cc.state

    mocklocs = []
    processed_mocks = []
    
    for mock in mocks:
        (segment, integration, dm, dt, amp, l0, m0) = mock

        dmind0 = np.abs((np.array(st.dmarr)-dm)).argmin()
        dtind0 = np.abs((np.array(st.dtarr)*st.inttime-dt)).argmin()
        integration0 = integration//st.dtarr[dtind0]
        dtarr = cc.state.dtarr
        time_ind = np.multiply(integration0, dtarr[dtind0])

        npixx = cc.state.npixx
        npixy = cc.state.npixy
        uvres = cc.state.uvres
        peakx_ind, peaky_ind = cc.state.calcpix(l0, m0, npixx, npixy,
                                                 uvres)

        processed_mocks.append([peakx_ind//downsample,
                                peaky_ind//downsample,
                                dmind0, time_ind])

        mocklocs.append((cc.segment, integration0, dmind0, dtind0, 0))
        
    return mocklocs, processed_mocks


def prep_to_cluster(cc, downsample=2):
    cc1 = deepcopy(cc)
    if len(cc1) > 1:
        logging.info(f'Pre-processing data of {len(cc)} candidates for clustering.')
        if downsample is None:
            downsample = cc1.prefs.cluster_downsampling

        candl = cc1.candl
        candm = cc1.candm
        npixx = cc1.state.npixx
        npixy = cc1.state.npixy
        uvres = cc1.state.uvres

        dmind = cc1.array['dmind']
        dtind = cc1.array['dtind']
        dtarr = cc1.state.dtarr
        timearr_ind = cc1.array['integration']  # time index of all the candidates

        time_ind = np.multiply(timearr_ind, np.array(dtarr).take(dtind))
        peakx_ind, peaky_ind = cc1.state.calcpix(candl, candm, npixx, npixy,
                                                 uvres)

        # stacking indices and taking at most one per bin
        data = np.transpose([peakx_ind//downsample,
                                       peaky_ind//downsample,
                                       dmind, time_ind])
    else:
        logging.info('Less than 2 candidates in candcollection. Not doing pre-processing.')
        data = None
    return data
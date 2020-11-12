from sklearn.manifold import TSNE
import pylab as plt
import numpy as np

def vis_tsne(data, dist_metric):
    tsne_res = TSNE(n_components=2,perplexity=30.0, early_exaggeration=12.0, 
         learning_rate=200.0, n_iter=1000, n_iter_without_progress=300, 
         min_grad_norm=1e-07, metric=dist_metric, init='random', 
         verbose=0, random_state=None, method='barnes_hut', angle=0.5).fit_transform(data)

#     tsne_one = tsne_res[:,0]
#     tsne_two = tsne_res[:,1]
    return tsne_res


def vis_cands(cc, title=None):
    candl = cc.candl
    candm = cc.candm
    npixx = cc.state.npixx
    npixy = cc.state.npixy
    uvres = cc.state.uvres
    dmind = cc.array['dmind']
    dtind = cc.array['dtind']
    dtarr = cc.state.dtarr
    timearr_ind = cc.array['integration']  # time index of all the candidates

    dms = np.array(cc.state.dmarr).take(dmind)    
    time_ind = np.multiply(timearr_ind, np.array(dtarr).take(dtind))
    peakx_ind, peaky_ind = cc.state.calcpix(candl, candm, npixx, npixy, uvres)

    snr = cc.snrtot
    
    i_segment = []
    i_integration = []
    i_dm = []
    i_dt = []
    i_amplitude = []
    i_l = []
    i_m = []

    for t in cc.state.prefs.simulated_transient:
        i_segment.append(t[0])
        i_integration.append(t[1])
        i_dm.append(t[2])
        i_dt.append(t[3])
        i_amplitude.append(t[4])
        i_l.append(t[5])
        i_m.append(t[6])    

    i_time = i_integration#*i_dt
    i_peakx_ind, i_peaky_ind = cc.state.calcpix(np.array(i_l), np.array(i_m), npixx, npixy, uvres)

    fig, ax = plt.subplots(2,2, figsize=(15, 12))

    ax[0,0].scatter(peakx_ind, peaky_ind, color = 'b', alpha=0.5, s=10*snr)
    ax[0,0].scatter(i_peakx_ind, i_peaky_ind, color='r', marker='+', s=20)
    ax[0,0].set_xlabel('l')
    ax[0,0].set_ylabel('m')
    ax[0,0].grid()

    ax[0,1].scatter(time_ind, dms, color = 'b', alpha=0.5, s=10*snr)
    ax[0,1].scatter(i_time, i_dm, color='r', marker='+', s=20)

    ax[0,1].set_xlabel('time')
    ax[0,1].set_ylabel('dm')
    ax[0,1].grid()


    ax[1,0].scatter(peakx_ind, dms, color = 'b', alpha=0.5, s=10*snr)
    ax[1,0].scatter(i_peakx_ind, i_dm, color='r', marker='+', s=20)

    ax[1,0].set_xlabel('l')
    ax[1,0].set_ylabel('dm')
    ax[1,0].grid()

    ax[1,1].scatter(peakx_ind, time_ind, color = 'b', alpha=0.5, s=10*snr)
    ax[1,1].scatter(i_peakx_ind, i_time, color='r', marker='+', s=20)

    ax[1,1].set_xlabel('l')
    ax[1,1].set_ylabel('time')
    ax[1,1].grid()
        
    plt.tight_layout()
    plt.show()
    return dms, time_ind, peakx_ind, peaky_ind

    
def plot_data(data, labels=[], snrs=[], tsne_res=None, title=None, save=False):
    import seaborn as sns
    from matplotlib import colors 

    #get a color palette with number of colors = number of clusters
    if len(labels):
        color_palette = sns.color_palette('deep', np.max(labels) + 1) 
        cluster_colors = [color_palette[x] if x >= 0
                          else (0.5, 0.5, 0.5)
                          for x in labels]    #assigning each cluster a color, and making a list

        cluster_colors = list(map(colors.rgb2hex, cluster_colors)) #converting sns colors to hex for bokeh
    else:
        cluster_colors = [(0.5, 0.5, 0.5) for x in range(data.shape[0])]

    if tsne_res:
        fig, ax = plt.subplots(3,2, figsize=(15, 12))
    else:
        fig, ax = plt.subplots(2,2, figsize=(15, 12))

    if len(snrs):
        snr = snrs
    else:
        snr = 10

    ax[0,0].scatter(data[:, 0], data[:, 1], color = cluster_colors, alpha=0.5, s=10*snr)
    ax[0,0].set_xlabel('l')
    ax[0,0].set_ylabel('m')
    ax[0,0].grid()

    ax[0,1].scatter(data[:, 2], data[:, 3], color = cluster_colors, alpha=0.5, s=10*snr)
    ax[0,1].set_xlabel('dm')
    ax[0,1].set_ylabel('time')
    ax[0,1].grid()

    ax[1,0].scatter(data[:, 0], data[:, 3], color = cluster_colors, alpha=0.5, s=10*snr)
    ax[1,0].set_xlabel('l')
    ax[1,0].set_ylabel('time')
    ax[1,0].grid()

    ax[1,1].scatter(data[:, 0], data[:, 2], color = cluster_colors, alpha=0.5, s=10*snr)
    ax[1,1].set_xlabel('l')
    ax[1,1].set_ylabel('dm')
    ax[1,1].grid()

    if tsne_res:
        ax[2,1].scatter(tsne_res[:,0], tsne_res[:,1], color = cluster_colors, s=10*snr)
        ax[2,1].set_xlabel('x')
        ax[2,1].set_ylabel('y')

        # ax[1,1].yaxis.tick_right()
        # ax[1,1].yaxis.set_label_position('right')
        ax[2,1].grid()

        ax[2,0].axis('off')        

    plt.tight_layout()
    plt.suptitle(title)
    if save:
        plt.savefig(title+'.png', bbox_inches='tight')
    plt.show()


def vis_clustering(data, cc, labels, tsne_res=None, title=None):

    import seaborn as sns
    from matplotlib import colors 
    
    #get a color palette with number of colors = number of clusters
    color_palette = sns.color_palette('deep', np.max(labels) + 1) 
    cluster_colors = [color_palette[x] if x >= 0
                      else (0.5, 0.5, 0.5)
                      for x in labels]    #assigning each cluster a color, and making a list

    cluster_colors = list(map(colors.rgb2hex, cluster_colors)) #converting sns colors to hex for bokeh

    candl = cc.candl
    candm = cc.candm
    npixx = cc.state.npixx
    npixy = cc.state.npixy
    uvres = cc.state.uvres
    dmind = cc.array['dmind']
    dtind = cc.array['dtind']
    dtarr = cc.state.dtarr
    timearr_ind = cc.array['integration']  # time index of all the candidates

    dms = np.array(cc.state.dmarr).take(dmind)    
    time_ind = np.multiply(timearr_ind, np.array(dtarr).take(dtind))
    peakx_ind, peaky_ind = cc.state.calcpix(candl, candm, npixx, npixy, uvres)

    snr = cc.snrtot
    
    #injected parameters
#     (i_segment, i_integration, i_dm, i_dt, i_amplitude, i_l, i_m) = cc.state.prefs.simulated_transient[0]
    i_segment = []
    i_integration = []
    i_dm = []
    i_dt = []
    i_amplitude = []
    i_l = []
    i_m = []

    for t in cc.state.prefs.simulated_transient:
        i_segment.append(t[0])
        i_integration.append(t[1])
        i_dm.append(t[2])
        i_dt.append(t[3])
        i_amplitude.append(t[4])
        i_l.append(t[5])
        i_m.append(t[6])    

    i_time = i_integration#*i_dt
    i_peakx_ind, i_peaky_ind = cc.state.calcpix(np.array(i_l), np.array(i_m), npixx, npixy, uvres)
    
    if tsne_res:
        fig, ax = plt.subplots(3,2, figsize=(15, 12))
    else:
        fig, ax = plt.subplots(2,2, figsize=(15, 12))

    ax[0,0].scatter(peakx_ind, peaky_ind, color = cluster_colors, alpha=0.5, s=10*snr)
    ax[0,0].scatter(i_peakx_ind, i_peaky_ind, color='r', marker='+', s=10)
    ax[0,0].set_xlabel('l')
    ax[0,0].set_ylabel('m')
    # ax[0,0].xaxis.tick_top()
    # ax[0,0].yaxis.tick_right()
    # ax[0,0].xaxis.set_label_position('top')
    # ax[0,0].yaxis.set_label_position('right')
    ax[0,0].grid()

    ax[0,1].scatter(time_ind, dms, color = cluster_colors, alpha=0.5, s=10*snr)
    ax[0,1].scatter(i_time, i_dm, color='r', marker='+', s=10)

    ax[0,1].set_xlabel('time')
    ax[0,1].set_ylabel('dm')
    # ax[0,1].xaxis.tick_top()
    # # ax9.xaxis.set_label_position('top')
    # ax[0,1].yaxis.tick_right()
    # ax[0,1].yaxis.set_label_position('right')
    ax[0,1].grid()

    ax[1,0].scatter(peakx_ind, dms, color = cluster_colors, alpha=0.5, s=10*snr)
    ax[1,0].scatter(i_peakx_ind, i_dm, color='r', marker='+', s=10)

    ax[1,0].set_xlabel('l')
    ax[1,0].set_ylabel('dm')
    # ax[1,0].yaxis.tick_right()
    # ax[1,0].yaxis.set_label_position('right')
    ax[1,0].grid()

    ax[1,1].scatter(peakx_ind, time_ind, color = cluster_colors, alpha=0.5, s=10*snr)
    ax[1,1].scatter(i_peakx_ind, i_time, color='r', marker='+', s=10)

    ax[1,1].set_xlabel('l')
    ax[1,1].set_ylabel('time')
    # ax[1,1].yaxis.tick_right()
    # ax[1,1].yaxis.set_label_position('right')
    ax[1,1].grid()

    if tsne_res:
        ax[2,1].scatter(tsne_res[:,0], tsne_res[:,1], color = cluster_colors, s=10*snr)
        ax[2,1].set_xlabel('x')
        ax[2,1].set_ylabel('y')

        # ax[1,1].yaxis.tick_right()
        # ax[1,1].yaxis.set_label_position('right')
        ax[2,1].grid()

        ax[2,0].axis('off')        
  
    plt.tight_layout()
    plt.suptitle(title)
    
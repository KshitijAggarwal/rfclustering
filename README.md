# rfclustering
Scripts used in clustering analysis of Realfast Candidates

* `run_search.py`: Runs a standard rfpipe transient search on an SDM and saves information of all the 
  candidates found. It does not perform any clustering. Uses `realfast_nocluster.yml` to set search 
  parameters. Requires [rfpipe](https://github.com/realfastvla/rfpipe).
* `inject_on_sim_data.py`: Script to inject one simulated FRB on simulated data and save unclustered candidates. It 
generated simulated data (based on randomly chosen VLA configuration and parameters) and injects an FRB in that data
  (again with randomly chosen transient parameters). Then it run the standard rfpipe search on it to detect 
  candidates from that transient and saves them without applying any clustering. Uses `realfast_nocluster.yml` to set search
  parameters. Requires [rfpipe](https://github.com/realfastvla/rfpipe).
* `random_hyperparameter_search.py`: Runs hyperparameter search for a given algorithm, on a given dataset. It 
  samples the hyperparameters randomly based on the functions in `util/hs_utils.py`. Requires 
  [sklean](https://scikit-learn.org/stable/) and [hdbscan](https://hdbscan.readthedocs.io/en/latest/index.html). 
* `utils`: Contains functions for data processing, plotting, clustering, hyperparameter search and metric calculation.  
* `notebooks/ncands_vs_snr.ipynb`: Notebook to estimate the number of candidates expected from a single pulse 
search given a DM list, boxcar list and observing configuration parameters. Realfast L-band parameters are currently
  implemented in the notebook. It uses Sec 2.3 of [this](https://ui.adsabs.harvard.edu/abs/2012PhDT.......306L/abstract) (also implemented 
  [here](https://github.com/thepetabyteproject/your/blob/master/your/utils/heimdall.py)) to generate the DM list. 


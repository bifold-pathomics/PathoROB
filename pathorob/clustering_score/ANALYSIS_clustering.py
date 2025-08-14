import sys
sys.path.append('../features/')
import data_manager
import utils_clustering
import numpy as np
import argparse
import random
import os
from itertools import combinations
from pathlib import Path
from sklearn.cluster import KMeans

### GET ARGUMENTS FROM COMMAND LINE ###########################################
parser = argparse.ArgumentParser()
# required
parser.add_argument(
    "--dataset", type=str, required=True,
    help="PathoROB dataset on which the clustering score is computed. Allowed options: camelyon, tolkach, tcga."
)
parser.add_argument(
    "--model", type=str, required=True,
    help="Name of foundation model (name of folder in feature_dir that holds the features of the model)."
)
# optional
parser.add_argument(
    "--features_dir", type=str, default='../data/',
    help="Optional. Path to directory that contains the features of the foundation model as subdirectory. Default: ../data/"
)
parser.add_argument(
    "--K", type=int,
    help="Optional. Number of clusters. If not fixed, K will be found by maximizing the silhouette score."
)
parser.add_argument(
    "--maxK", type=int, default=30,
    help="Optional. Upper endpoint of the interval of values tested for K. Default: 30."
)
parser.add_argument(
    "--metadata_dir", type=str, default='../data/metadata/',
    help="Optional. Path to directory that contains the metadata files. Default: ../data/metadata/"
)
parser.add_argument(
    "--metric", type=str, default='cosine',
    help="Optional. Distance metric. Allowed options: cosine, euclidean. Default: cosine."
)
parser.add_argument(
    "--minK", type=int, default=2,
    help="Optional. Lower endpoint of the interval of values tested for K. Default: 2."
)
parser.add_argument(
    "--num_trials", type=int, default=50,
    help="Optional. Number of repetitions to run the clustering (for calculating the average clustering score). Default: 50."
)
parser.add_argument(
    "--reset_K", type=bool, default=False,
    help="Optional. Forces to find optimal number of clusters again, even though there might be saved results. Default: False."
)
parser.add_argument(
    "--results_dir", type=str, default='../results/clustering_score/',
    help="Optional. Path to directory to save the clustering score results. Default: ../results/clustering_score/"
)

args            = parser.parse_args()
# required
dataset         = args.dataset
model           = args.model
# optional
features_dir    = args.features_dir
K               = args.K
maxK            = args.maxK
metadata_dir    = args.metadata_dir
metric          = args.metric
minK            = args.minK
num_trials      = args.num_trials
reset_K         = args.reset_K
results_dir     = args.results_dir

if features_dir[-1] !='/': features_dir = features_dir + '/'
features_dir = features_dir + model + '/'
if not(os.path.isdir( features_dir )): 
    raise ValueError(f"{features_dir} : features_dir not found.")
if not(os.path.isdir( metadata_dir )): 
    raise ValueError(f"{metadata_dir} : metadata_dir not found.")
if (K is not None) and K<2:
    raise ValueError(f"{K} : Number of clusters, K, must be >= 2.")
if minK<2 or minK>maxK: 
    raise ValueError(f"{minK} : Lower endpoint, minK, must be >= 2 and smaller than {maxK} (maxK).")
if maxK<2 or minK>maxK: 
    raise ValueError(f"{maxK} : Upper endpoint, maxK, must be >= 2 and larger than {minK} (minK).")
if not((metric == 'cosine') or (metric == ' euclidean')):
    raise ValueError("No valid metric argument. Allowed options: cosine, euclidean")
if num_trials<1:
    raise ValueError(f"{num_trials} : Number of trials, num_trials, must be at least 1.")
if results_dir != '../results/clustering_score/':
    if results_dir[-1] !='/': results_dir = results_dir + '/'
else:
    results_dir = results_dir + model + '/' + dataset + '/'
Path(results_dir).mkdir(parents=True, exist_ok=True)
###############################################################################

### PREPARE DATA LOADER #######################################################
# initialize data manager
data_loader = data_manager.FeatureDataManager(features_dir, metadata_dir=metadata_dir)
if not(dataset in data_loader.get_available_datasets()): 
    raise ValueError(f"{dataset} : unrecognized dataset.")
###############################################################################

### LOAD METADATA #############################################################
# load the metadata corresponding to the dataset
if dataset=='camelyon':
    metadata_name = 'camelyon_reduced'
elif dataset=='tolkach':
    metadata_name = 'tolkach_esca_reduced'
elif dataset=='tcga':
    metadata_name = 'tcga_4x4'
else:
    raise ValueError(f"For dataset {dataset} no metadata known.")
metadata = data_loader.load_metadata(metadata_name)

# get the possible options of the bio and mc labels
bio_options   = metadata['biological_class'].unique()
mc_options    = metadata['medical_center'  ].unique()
# construct all pairings of bio and mc and the corresponding entries in metadata 
combis, metadata_per_combi = utils_clustering.get_metadata_per_combi(bio_options, 
                                                               mc_options, 
                                                               metadata)
###############################################################################

print(f'Clustering score evaluation for {features_dir}', flush=True)
random.seed(1000)
# go over all possible combinations of two bio classes x two mc classes
# which then forms the data for one individual experiment
for bio1, bio2 in combinations(bio_options, r=2):
    for mc1, mc2 in combinations(mc_options, r=2):
        cc_str = f'{bio1}-{bio2}-{mc1}-{mc2}'
        current_combis = [f'{bio1}-{mc1}',
                          f'{bio1}-{mc2}',
                          f'{bio2}-{mc1}',
                          f'{bio2}-{mc2}']
        # check whether each of these combinations exist in metadata
        if not(all([(cc in combis) for cc in current_combis])):
            print(f'Warning: {current_combis} not found in metadata. Skip.', flush=True)
            continue
        ### LOAD DATA #########################################################
        print(cc_str, flush=True)
        Z = []
        bio_labels = []
        mc_labels = []
        for cc in current_combis:
            loaded_data = data_loader.load_features(dataset, metadata_per_combi[cc])
            Z.extend(loaded_data)
            bio_labels.extend(metadata_per_combi[cc]['biological_class'].to_list())
            mc_labels.extend( metadata_per_combi[cc]['medical_center'  ].to_list())
        # convert to array so that it is easier to handle
        Z          = np.array(Z)
        bio_labels = np.array(bio_labels)
        mc_labels  = np.array(mc_labels)
        
        # normalize data for kmeans based on cosine distance
        if metric == 'cosine':
            # normalize to unit length
            Z_norm = Z / np.sqrt((Z**2).sum(axis=1))[:,None]
        else:
            Z_norm = Z
        ###########################################################################
        
        ### SELECT K ##############################################################
        find_K = True
        if K is None:
            if not(reset_K) and os.path.isfile(f'{results_dir}{cc_str}_SilhouetteScores.csv'):
                silhouette_scores = np.genfromtxt(f'{results_dir}{cc_str}_SilhouetteScores.csv', delimiter=',')
                tested_minK = silhouette_scores[0, 0]
                tested_maxK = silhouette_scores[0,-1]
                if tested_minK<=minK and tested_maxK>=maxK:
                    find_K = False
                    K = int(silhouette_scores[0, np.argmax(silhouette_scores[1,:])])
                    print(f"... loading optimal number of clusters from file: [{tested_minK}, {tested_maxK}] -> K = {K}.", flush=True)
            if find_K:
                Ks = np.linspace(minK, maxK, maxK-minK+1, dtype=int)
                silhouette_scores = utils_clustering.compute_silhouette_score_K(Z_norm, Ks, metric, 1337)
                K  = Ks[np.argmax(silhouette_scores)]
                print(f"... optimal number of clusters: [{minK}, {maxK}] -> K = {K}.", flush=True)
                # save to filesystem for potential later use
                np.savetxt(f'{results_dir}{cc_str}_SilhouetteScores.csv', [Ks, silhouette_scores], delimiter=',')
        ###########################################################################
        
        Aris = np.zeros([num_trials, 3])
        for t in range(num_trials):
            rnd_seed = 1337 + t
            ### CLUSTERING WITH OPT. K ################################################
            kmeans = KMeans(n_clusters=K, random_state=rnd_seed, n_init=5, init='random').fit(Z_norm)
            cluster_labels = kmeans.labels_
            ###########################################################################
            
            ### CLUSTERING METRICS ####################################################
            ari_bio, ari_mc, cs = utils_clustering.compute_clustering_score(bio_labels, 
                                                                      mc_labels, 
                                                                      cluster_labels)
            Aris[t, :] = [ari_bio, ari_mc, cs]
            # save to filesystem for later use
            np.savetxt(f'{results_dir}{cc_str}_ARI.csv', Aris, delimiter=',')
            ###########################################################################
        print(f"... average clustering score: {np.mean(Aris[:,-1])}", flush=True)
        print(f"... standard deviation:       {np.std( Aris[:,-1])}", flush=True)
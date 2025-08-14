import numpy as np
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn import metrics

### DATA PREPARATION ##########################################################
def get_2x2_combis(bio_options, confounder_options):
    """
    Constructs all 2x2 pairings of the biological and confounder labels.

    Parameters
    ----------
    bio_options :  bio_label_type list
        list of all possible biological labels.
    confounder_options :  confounder_label_type list
        list of all possible confounder labels.

    Returns
    -------
    combis : string set
        set of all combinations of bio_options and confounder_options as string.

    """
    combis = set()
    for bio1, bio2 in combinations(bio_options, r=2):
        for conf1, conf2 in combinations(confounder_options, r=2):
            combis.update([f'{bio1}-{conf1}',
                        f'{bio1}-{conf2}',
                        f'{bio2}-{conf1}',
                        f'{bio2}-{conf2}'])
    return combis

def get_metadata_per_combi(bio_options, confounder_options, full_metadata):
    """
    Finds all entries of the metadata file corresponding to a combination of a 
    bio and confounder label.

    Parameters
    ----------
    combis : string set
        set of all combinations of bio_options and confounder_options as string.
    metadata_entries : pandas DataFrame
        content of metadata file.

    Returns
    -------
    idx_per_combi : dictionary (string - int list)
        idx_per_combi[bio-conf] gives a list of indices of all entries in meta
        data file corresponding to the bio-conf combi.

    """
    combis             = set()
    metadata_per_combi = {} 
    for bio in bio_options:
        for conf in confounder_options:
            combi = f'{bio}-{conf}'
            sub_metadata = full_metadata.loc[(full_metadata['biological_class']==bio)&
                                             (full_metadata['medical_center']==conf)]
            if len(sub_metadata)>0:
                metadata_per_combi[combi] = sub_metadata
                combis.add(combi)
    return combis, metadata_per_combi
###############################################################################

### SELECTION OF NR. OF CLUSTERS, K ###########################################
def compute_silhouette_score_K(X, Ks, metric, rnd_state):
    """
    Computes the silhouette score of a K-means clustering for all passed values
    of K. 

    Parameters
    ----------
    X : (N,D) float array
        N data samples in D dimensions. If metric='cosine' X is expected to be 
        normalized.
    Ks : (num_K,) int array
        number of clusters (K) for the K-means clustering.
    metric : string
        metric to be used to compute the clustering. Valid options are:
        ‘cosine’, ‘euclidean’.
        If ‘cosine’, the data is normalized to unit length so that regular 
        K-means clustering can be applied.
    rnd_state: int
        random seed for initializing the k-means algorithm

    Returns
    -------
    SS : (num_K,) float array
        silhouette scores for every value of K.

    """
    SS = np.zeros(len(Ks))
    for i,k in enumerate(Ks):
        kmeans = KMeans(n_clusters=k, random_state=rnd_state, init="random", n_init=20).fit(X)
        cluster_labels = kmeans.labels_
        SS[i] = metrics.silhouette_score(X, cluster_labels, metric=metric)
    return SS
###############################################################################

### CLUSTERING SCORE ##########################################################
def compute_clustering_score(bio_labels, confounder_labels, cluster_labels):
    """
    Computes the clustering score and the adjusted rand indices for the 
    biological and confounder labels.

    Parameters
    ----------
    bio_labels : (N,) bio_label_type array
        biological labels of the data points.
    confounder_labels : (N,) confounder_label_type array
        confounder labels of the data points.
    cluster_labels : (N,) int array
        labels returned by clustering algorithm.


    Returns
    -------
    ari_bio : float
        adjusted rand index for the biological labels.
    ari_conf : float
        adjusted rand index for the confounding labels.
    cs : float
        clustering score.

    """
    ari_bio  = metrics.adjusted_rand_score(bio_labels,        cluster_labels)
    ari_conf = metrics.adjusted_rand_score(confounder_labels, cluster_labels)
    cs       = ari_bio - ari_conf
    return ari_bio, ari_conf, cs
###############################################################################
import copy
import json
import os
import pickle
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import mode

class OutputFiles(StrEnum):
    SUMMARY_FILE = "results_summary.json"
    BALANCED_ACCURACY = "bal-acc-bio"
    FREQUENCIES = "frequencies-same-class"


def bootstrapped_robustness_index(SO_cum, OS_cum, n_bootstrap = 1000):
    nr_samples = SO_cum.shape[0]
    nr_neighbors = SO_cum.shape[1]  # number of k nearest neighbors

    row_index = np.arange(nr_samples)

    bootstrapped_values = np.empty((n_bootstrap, nr_neighbors))
    for i in range(n_bootstrap):
        sample = np.random.choice(row_index, size=nr_samples, replace=True)
        sample_SO = SO_cum[sample, :]
        sample_OS = OS_cum[sample, :]
        nr_SO = np.sum(sample_SO, axis=0)  # number of k nearest neighbors that have the same bio class and other confounding class
        nr_OS = np.sum(sample_OS, axis=0)  # number of k nearest neighbors that have the other bio class and same confounding class
        total = nr_SO + nr_OS
        robustness_index = nr_SO / total  # robustness index for same bio class, other conf
        bootstrapped_values[i,:] = robustness_index

    bootstrapped_mean = np.mean(bootstrapped_values, axis=0)  # mean over all bootstrapped samples
    bootstrapped_std = np.std(bootstrapped_values, axis=0, ddof=1)

    return bootstrapped_mean, bootstrapped_std


def restrict_to_same_nr_cols(results, key):
    """
    Given a 1D or 2D list of results, restrict them to the same number of columns.
    """
    nr_cols = [r.shape[-1] for r in results]
    min_nr_cols = np.min(nr_cols)
    if len(np.unique(nr_cols)) > 1:
        print(f"found these unique nr columns for {key}: {np.unique(nr_cols)}. Reducing all to {min_nr_cols}.")
    if len(results[0].shape) == 1:  # if results are 1D arrays, convert to 2D
        results = [r[:min_nr_cols] for r in results]
    elif len(results[0].shape) == 2:  # if results are 2D arrays, reduce columns
        results = [r[:, :min_nr_cols] for r in results]
    else:
        raise ValueError(f"unhandled case: 2+D array for key {key}")
    return results

def get_matrix(total_stats, mat_name):
    mat = list(total_stats[mat_name].values())
    mat = restrict_to_same_nr_cols(mat, mat_name)
    mat = np.vstack(mat)
    return mat

def get_cumulative_sum(mat):
    mat_cum = np.cumsum(mat, axis=1)
    mat_cum_colsums = np.sum(mat_cum, axis=0)
    return mat_cum, mat_cum_colsums

def compute_generalization_index(total_stats):
    SS = total_stats["fraction_SS-cum-norm"]
    SO = total_stats["fraction_SO-cum-norm"]
    OS = total_stats["fraction_OS-cum-norm"]
    OO = total_stats["fraction_OO-cum-norm"]

    SSSO = SS * SO
    SOOS = SO * OS
    SSOO = SS * OO

    generalization_index = (SSSO + SOOS) / (SSSO + SSOO)
    return generalization_index

def compute_OOD_performance(total_stats):
    SO = total_stats["fraction_SO-cum-norm"]
    OO = total_stats["fraction_OO-cum-norm"]

    ood_performance = SO / (SO + OO)
    return ood_performance

def compute_ID_performance(total_stats):
    SS = total_stats["fraction_SS-cum-norm"]
    OS = total_stats["fraction_OS-cum-norm"]

    id_performance = SS / (SS + OS)
    return id_performance

def compute_SO_SS_ratio(total_stats):
    SO = total_stats["fraction_SO-cum-norm"]
    SS = total_stats["fraction_SS-cum-norm"]

    SO_SS_rati = SO / SS
    return SO_SS_rati


def aggregate_stats(all_stats, compute_bootstrapped_robustness_index=True):
    """

    Parameters
    ----------
    all_stats
    compute_bootstrapped_robustness_index: optional. Used to obtain an estimate of the standard deviation. This is
    time consuming; can be skipped if only the robustness index itself is required; this is available as `robustness_index`

    Returns
    -------

    """
    keys = set()
    key_type = {}
    for s in all_stats:
        new_keys = [k for k in s.keys() if k not in keys]
        for k in new_keys:
            key_type[k] = type(s[k])
        keys.update(new_keys)
    max_k_per_fold = [int(np.max(all_stats[i]["k"])) for i in range(len(all_stats))]
    max_k = np.min(max_k_per_fold)
    print(f"aggregate_stats: restricting max_k to {max_k}. Found max_k_per_fold {max_k_per_fold}.")
    total_stats = {}
    total_stats["max_k"] = max_k

    for key in keys:
        if key == "k":
            k_values = all_stats[0]["k"]
            total_stats["k"] = [int(k) for k in k_values if k <= max_k]  # restrict k values to max_k
        elif key == "nr_samples":
            total_stats[key] = np.mean([stats[key] for stats in all_stats])
        elif key in ["SS", "SO", "OS", "OO"]:
            total_stats[key] = {}
            for stat in all_stats:
                total_stats[key].update(stat[key]) #update with all SOs
        elif key_type[key] is np.ndarray:
            raise ValueError("this should not happen.")
        elif key_type[key] is dict:
            total_stats[key] = {}
            results = []
            for stats in all_stats:
                if key in stats:
                    results.append(stats[key])
            avg_results = {}
            for k in results[0].keys():
                avg_results[k] = []
                for r in results:
                    avg_results[k].append(r[k])
                total_stats[key][k] = np.vstack(avg_results[k]).mean(axis=0)
        else:
            raise ValueError("tbd unhandled key (stats per class)", key, "type", type(all_stats[0][key]))

    SS = get_matrix(total_stats, "SS")
    SO = get_matrix(total_stats, "SO")
    OS = get_matrix(total_stats, "OS")
    OO = get_matrix(total_stats, "OO")

    total_stats.pop("SS", None) #full matrix no longer needed; keep only aggregated forms below
    total_stats.pop("SO", None)
    total_stats.pop("OS", None)
    total_stats.pop("OO", None)

    nr_SS = np.sum(SS, axis=0)
    nr_SO = np.sum(SO, axis=0)
    nr_OS = np.sum(OS, axis=0)
    nr_OO = np.sum(OO, axis=0)

    SS_cum, nr_ss_cum = get_cumulative_sum(SS)
    SO_cum, nr_so_cum = get_cumulative_sum(SO)
    OS_cum, nr_os_cum = get_cumulative_sum(OS)
    OO_cum, nr_oo_cum = get_cumulative_sum(OO)

    total = nr_SS + nr_SO + nr_OS + nr_OO
    total_cum = nr_ss_cum + nr_so_cum + nr_os_cum + nr_oo_cum

    total_stats["fraction_SS-norm"] = nr_SS / total
    total_stats["fraction_SO-norm"] = nr_SO / total
    total_stats["fraction_OS-norm"] = nr_OS / total
    total_stats["fraction_OO-norm"] = nr_OO / total
    total_stats["fraction_SS-cum-norm"] = nr_ss_cum / total_cum
    total_stats["fraction_SO-cum-norm"] = nr_so_cum / total_cum
    total_stats["fraction_OS-cum-norm"] = nr_os_cum / total_cum
    total_stats["fraction_OO-cum-norm"] = nr_oo_cum / total_cum

    total_stats["robustness_index"] = total_stats["fraction_SO-cum-norm"] / (total_stats["fraction_SO-cum-norm"] + total_stats["fraction_OS-cum-norm"])
    total_stats["generalization_index"] = compute_generalization_index(total_stats)
    total_stats["OOD_performance"] = compute_OOD_performance(total_stats)
    total_stats["ID_performance"] = compute_ID_performance(total_stats)
    total_stats["SO_SS_ratio"] = compute_SO_SS_ratio(total_stats)

    #optionally use bootstrapping to get std dev estimate
    if compute_bootstrapped_robustness_index:
        print("bootstrapping robustness index")
        bootstrapped_mean, bootstrapped_std = bootstrapped_robustness_index(SO_cum, OS_cum)
        total_stats["robustness_index-mean"] = bootstrapped_mean
        total_stats["robustness_index-std"] = bootstrapped_std
        print(f"computed bootstrapped robustness_index-mean {total_stats['robustness_index-mean'][:3]} std {total_stats['robustness_index-std'][:3]}")
    else:
        total_stats["robustness_index-mean"] = [-1]
        total_stats["robustness_index-std"] = [-1]
    return total_stats

def calculate_fractions(same_bio_class, same_conf_class, compute_fractions):
    SS = same_bio_class & same_conf_class #SB-SC
    SO = same_bio_class & (1 - same_conf_class) #OB-SC
    OS = (1 - same_bio_class) & same_conf_class #SB-OC
    OO = (1 - same_bio_class) & (1 - same_conf_class) #OB-OC

    nr_samples = SS.shape[0]
    fraction_SS = np.sum(SS, axis=0) / nr_samples #fraction of neighbors that have the same biological and confounding class
    fraction_SO = np.sum(SO, axis=0) / nr_samples #every val split has the same size, so using fractions here is equivalent to using total counts
    fraction_OS = np.sum(OS, axis=0) / nr_samples
    fraction_OO = np.sum(OO, axis=0) / nr_samples

    fraction_SS_cum = fraction_SS.cumsum()
    fraction_SO_cum = fraction_SO.cumsum()
    fraction_OS_cum = fraction_OS.cumsum()
    fraction_OO_cum = fraction_OO.cumsum()

    robustness_index = fraction_SO_cum / (fraction_SO_cum + fraction_OS_cum) #robustness index for same bio class, other conf

    total = fraction_SO_cum + fraction_OS_cum
    fraction_SO_cum_norm = fraction_SO_cum / total
    fraction_OS_cum_norm = fraction_OS_cum / total

    if compute_fractions:
        fractions = {
            "fraction_SS": fraction_SS, #SB-SC, i.e. fraction of neighbors with same biological and confounding class for each k
            "fraction_SO": fraction_SO, #OB-SC
            "fraction_OS": fraction_OS, #SB-OC
            "fraction_OO": fraction_OO, #OB-OC
            "fraction_SS-cum" : fraction_SS_cum,
            "fraction_SO-cum" : fraction_SO_cum,
            "fraction_OS-cum" : fraction_OS_cum,
            "fraction_OO-cum" : fraction_OO_cum,
            "fraction_SO-cum-norm" : fraction_SO_cum_norm, #normalized such that SO and OS sum to 1
            "fraction_OS-cum-norm" : fraction_OS_cum_norm,
            "robustness_index_tmp": robustness_index, #will be calculated with bootstrapping later to get mean and std
        }
    else:
        fractions = None
    return fractions, SS, SO, OS, OO

def calculate_fractions_per_cat(same_bio_class, same_conf_class, df_test, biological_class_field, confounding_class_field):
    fractions_per_class = {}
    bio_classes = np.unique(df_test[biological_class_field].values)
    conf_classes = np.unique(df_test[confounding_class_field].values)
    if len(np.unique(list(bio_classes) + list(conf_classes))) != len(bio_classes) + len(conf_classes):
        raise ValueError(f"biological classes {bio_classes} and confounding classes {conf_classes} overlap. Please use different field names for these classes.")
    for bio_class in bio_classes:
        same_bio_class_sel = same_bio_class[df_test[biological_class_field].values == bio_class]
        same_conf_class_sel = same_conf_class[df_test[biological_class_field].values == bio_class]
        fractions_per_class[bio_class], _, _, _, _ = calculate_fractions(same_bio_class_sel, same_conf_class_sel, compute_fractions=True)
    for conf_class in conf_classes:
        same_bio_class_sel = same_bio_class[df_test[confounding_class_field].values == conf_class]
        same_conf_class_sel = same_conf_class[df_test[confounding_class_field].values == conf_class]
        fractions_per_class[conf_class], _, _, _, _ = calculate_fractions(same_bio_class_sel, same_conf_class_sel, compute_fractions=True)
    return fractions_per_class

def calculate_fraction_same_class_per_k_value(df: pd.DataFrame, neighbor_index_mat: np.ndarray,
                                              biological_class_field: str = "cancer_type",
                                              confounding_class_field: str = "medical_center",
                                              ) -> dict:
    """
    For a range of neighbor distances k (k_th neighbor), calculate:
       a) the fraction of neighbors that have the same biological class
       b) the fraction of neighbors that have the same confounding class

    Parameters
    ----------
    df_embeddings_with_meta: pd.DataFrame with columns embedding, biological_class_field, confounding_class_field
    neighbor_index_mat: np.ndarray with indices of neighbors for each sample
    query_mask: np.ndarray with boolean mask for query samples: samples for which to analyze neighbors
    num_workers: int number of workers to use for parallel processing
    biological_class_field: field name for the biological class
    confounding_class_field: field name for the confounding class
    min_neighbor_index: min number of neighbors to consider
    max_neighbor_index: max number of neighbors to consider

    Returns: dict with results
    -------
    """

    nr_samples = len(df)
    nr_neighbors = neighbor_index_mat.shape[1]

    same_bio_class = has_same_class(df, neighbor_index_mat, nr_samples, biological_class_field)
    same_conf_class = has_same_class(df, neighbor_index_mat, nr_samples, confounding_class_field)

    result = {
        "k": np.arange(1, nr_neighbors + 1),
        "nr_samples": len(neighbor_index_mat),
    }

    _, SS, SO, OS, OO = calculate_fractions(same_bio_class, same_conf_class, compute_fractions=False)

    fractions = {}
    fractions["SS"] = {}
    fractions["SO"] = {}
    fractions["OS"] = {}
    fractions["OO"] = {}
    for row_index, patch_name in enumerate(df.patch_name.values):
        fractions["SS"][patch_name] = SS[row_index,:]
        fractions["SO"][patch_name] = SO[row_index,:]
        fractions["OS"][patch_name] = OS[row_index,:]
        fractions["OO"][patch_name] = OO[row_index,:]

    result.update(fractions)

    fractions_per_cat = calculate_fractions_per_cat(same_bio_class, same_conf_class, df, biological_class_field, confounding_class_field)
    result.update(fractions_per_cat)

    return result

def get_other_class(c, classes):
    c1 = classes[0]
    c2 = classes[1]
    if len(classes) > 2:
        raise ValueError('len 2 expected')
    if c == c1:
        return c2
    elif c == c2:
        return c1
    else:
        raise ValueError('c not in classes')

def calculate_all_frequencies_same_class_per_neighbor_distance(df_embeddings_with_meta: pd.DataFrame, neighbor_index_mat: np.ndarray,
                                                         bio_class_query, conf_class_query, bio_classes, conf_classes,
                                                         biological_class_field: str = "cancer_type",
                                                         confounding_class_field: str = "medical_center",
                                                         min_neighbor_index: int = 1, max_neighbor_index: int = -1,
                                                         ) -> dict:
    """
    For a range of neighbor distances k (k_th neighbor), calculate:
       a) the fraction of neighbors that have the same biological class
       b) the fraction of neighbors that have the same confounding class

    Parameters
    ----------
    df_embeddings_with_meta: pd.DataFrame with columns embedding, biological_class_field, confounding_class_field
    neighbor_index_mat: np.ndarray with indices of neighbors for each sample
    query_mask: np.ndarray with boolean mask for query samples: samples for which to analyze neighbors
    holdout_mask: np.ndarray with boolean mask for holdout samples, to be excluded from the analysis
    num_workers: int number of workers to use for parallel processing
    biological_class_field: field name for the biological class
    confounding_class_field: field name for the confounding class
    min_neighbor_index: min number of neighbors to consider
    max_neighbor_index: max number of neighbors to consider

    Returns: dict with results
    -------
    """

    other_bio_class = [get_other_class(c, bio_classes) for c in bio_class_query]
    other_conf_class = [get_other_class(c, conf_classes) for c in conf_class_query]

    nr_samples = len(df_embeddings_with_meta)
    if max_neighbor_index > 1:
        n_neighbors = min(max_neighbor_index, nr_samples)
    else:
        n_neighbors = nr_samples

    freq_same_bio_class_same_conf, avg_freq_same_bio_class_same_conf, k_values = calculate_avg_frequency_same_classes(
        df_embeddings_with_meta, bio_class_query, conf_class_query, neighbor_index_mat, nr_samples, n_neighbors,
        biological_class_field, confounding_class_field, min_neighbor_index, max_neighbor_index
    )
    freq_same_bio_class_other_conf, avg_freq_same_bio_class_other_conf, k_values = calculate_avg_frequency_same_classes(
        df_embeddings_with_meta, bio_class_query, other_conf_class, neighbor_index_mat, nr_samples, n_neighbors,
        biological_class_field, confounding_class_field, min_neighbor_index, max_neighbor_index
    )
    freq_other_bio_class_same_conf, avg_freq_other_bio_class_same_conf, k_values = calculate_avg_frequency_same_classes(
        df_embeddings_with_meta, other_bio_class, conf_class_query, neighbor_index_mat, nr_samples, n_neighbors,
        biological_class_field, confounding_class_field, min_neighbor_index, max_neighbor_index
    )
    freq_other_bio_class_other_conf, avg_freq_other_bio_class_other_conf, k_values = calculate_avg_frequency_same_classes(
        df_embeddings_with_meta, other_bio_class, other_conf_class, neighbor_index_mat, nr_samples, n_neighbors,
        biological_class_field, confounding_class_field, min_neighbor_index, max_neighbor_index
    )

    result = {
        "k": k_values,
        "nr_samples": len(neighbor_index_mat),
        "freq_same_bio_class_same_conf": freq_same_bio_class_same_conf, #given that SB-SC and OB-OC are excluded, freq_same_bio_class and freq_same_conf_class sum to one
        "freq_same_bio_class_other_conf": freq_same_bio_class_other_conf,
        "freq_other_bio_class_same_conf": freq_other_bio_class_same_conf,
        "freq_other_bio_class_other_conf": freq_other_bio_class_other_conf,
    }
    return result


def get_field_names_given_dataset(dataset):
    return "biological_class", "medical_center"


def get_combi_meta_info(meta, patch_names, embeddings, combi):
    index_combi = np.where(meta.subset.values==combi)[0]
    meta_combi = meta.iloc[index_combi,:].reset_index(drop=True)
    combi_patches = set(meta_combi.patch_name.values)
    meta_combi.set_index("patch_name",inplace=True)
    meta_combi["embedding"] = [np.zeros(embeddings.shape[1]) for _ in range(len(meta_combi))]
    sel = [p in combi_patches for p in patch_names]
    if np.sum(sel) == 0:
        print(f'no patches for combi {combi}')
        return None
    patch_names_sel = patch_names[sel]
    embeddings_sel = embeddings[sel,:]
    for k, patch_name in enumerate(patch_names_sel):
        meta_combi.at[patch_name, 'embedding'] = embeddings_sel[k, :]
    meta_combi.reset_index(inplace=True)

    embedding_sums = np.abs(np.vstack(meta_combi.embedding.values)).sum(axis=1)
    nr_zero_embeddings = np.sum(embedding_sums == 0)
    if nr_zero_embeddings > 0:
        raise ValueError(f"nr zero embeddings {nr_zero_embeddings} for combi {combi}")

    return meta_combi

def do_knn_checks(X_train, X_test):
    nr_unique_values_x_train = len(np.unique(X_train.flatten()))
    if nr_unique_values_x_train == 1:
        raise ValueError("X_train only has 1 value")
        return 0

    nr_unique_values_x_test = len(np.unique(X_test.flatten()))
    if nr_unique_values_x_test == 1:
        raise ValueError("X_test only has 1 value")
        return 0

def filter_out_query_case_from_neighbors(meta, dataset, knn_indices, X_train, X_test):
    group_id = meta["slide_id"].values

    min_nr_neighbors = len(X_train)
    result = []
    for query_sample_index in range(len(X_test)):
        query_group_id = group_id[query_sample_index]
        # filter out the query case from the neighbors
        neighbor_subset = [i for i in knn_indices[query_sample_index] if group_id[i] != query_group_id]
        result.append(neighbor_subset)
        if len(neighbor_subset) == 0:
            print(f"filter_out_query_case_from_neighbors: warning: 0 neighbors! all {len(knn_indices[query_sample_index])} nearest neighbors are from case ID {query_group_id}. "
                  f"Consider increasing max_k in get_max_k() for the {dataset} dataset.")
        if len(neighbor_subset) < min_nr_neighbors:
            min_nr_neighbors = len(neighbor_subset)
    print(f"Filtered out query case from knn neighbors. Original #neighbors: {knn_indices.shape[1]} filtered: {min_nr_neighbors}")
    result = [row[:min_nr_neighbors] for row in result]
    knn_indices = np.array(result)
    return knn_indices

def evaluate_knn_accuracy(meta, dataset, X_train, X_test, y_train, y_test, n_neighbors, num_workers, knn_distances=None, knn_indices=None, do_checks=False):
    knn_model = KNeighborsClassifier(
        n_neighbors=n_neighbors, n_jobs=num_workers
    )

    if do_checks:
        do_knn_checks(X_train, X_test)

    knn_fitted = False
    if knn_distances is None:
        knn_model.fit(X_train, y_train)
        knn_fitted = True
        knn_distances, knn_indices  = knn_model.kneighbors(X_test, return_distance=True)
        knn_indices = filter_out_query_case_from_neighbors(meta, dataset, knn_indices, X_train, X_test)

    knn_indices_sel = knn_indices[:, :n_neighbors]
    effective_n_neighbors = knn_indices_sel.shape[1] #actual number of neighbors used for prediction, after filtering out the query case
    neighbor_labels = y_train[knn_indices_sel]
    y_pred = mode(neighbor_labels, axis=1).mode.flatten()
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

    report_auc_per_class = False #can be set to False for speedup
    auc_per_class = None
    num_classes = len(np.unique(y_test))
    if num_classes > 2 and report_auc_per_class:
        if not knn_fitted:
            knn_model.fit(X_train, y_train)
        y_pred_proba = knn_model.predict_proba(X_test)
        auc_per_class = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average=None)

    return balanced_accuracy, auc_per_class, knn_distances, knn_indices, effective_n_neighbors


def get_k_values(dataset, paired_evaluation, opt_k=None, max_samples_per_group=0):
    if opt_k and opt_k > 0:
        margin = max_samples_per_group
        k_values = np.array([opt_k + margin])
        return k_values
    else:
        max_k = get_max_k(dataset, paired_evaluation)
        k_values = np.array([1, 3, 5, 7, 9] + list(np.arange(11, max_k, 10)))
        return k_values

def calculate_avg_frequency_same_classes(df_embeddings_with_meta: pd.DataFrame,
                                       class_sample1: np.ndarray,
                                       class_sample2: np.ndarray,
                                       neighbor_index_mat: np.ndarray, nr_samples: int,
                                       n_neighbors: int, class_name1: str, class_name2: str,min_neighbor_index: int,
                                       max_neighbor_index: int) -> float:
    freq_same_class, k_values = calculate_fraction_same_classes(df_embeddings_with_meta, class_sample1, class_sample2, neighbor_index_mat, nr_samples, n_neighbors, class_name1, class_name2)
    avg_freq_same_class = float(np.mean(freq_same_class[min_neighbor_index:max_neighbor_index]))
    return freq_same_class, avg_freq_same_class, k_values

def get_max_k(dataset, paired_evaluation):
    if dataset in ["camelyon"]:
        max_k = 600
    elif dataset in ["tcga"]:
        if paired_evaluation:
            max_k = 1200
        else:
            max_k = 600
    elif dataset in ["tolkach_esca"]:
        max_k = 1000
    else:
        raise ValueError(f"please provide max k for {dataset}")
    return max_k


def has_same_class(df: pd.DataFrame,
                   neighbor_index_mat: np.ndarray, nr_samples: int,
                   class_name: str) -> np.ndarray:
    """
    For each value of k up to n_neighbors, calculate the number of samples for which the kth neighbor has the same
    class as the sample.
    """

    same_class = np.zeros_like(neighbor_index_mat)
    nr_samples = neighbor_index_mat.shape[0]
    for sample_index in range(nr_samples):
        neighbor_k = neighbor_index_mat[sample_index,:] #get neighbors of sample
        class_neighbor_k = df.iloc[neighbor_k][class_name].values
        class_sample = df.iloc[sample_index][class_name] #get class of sample
        neighbor_has_same_class = class_neighbor_k == class_sample
        same_class[sample_index,:] = neighbor_has_same_class #store in same_class

    return same_class

def calculate_fraction_same_classes(df_embeddings_with_meta: pd.DataFrame,
                                  class1_sample, class2_sample,
                                  neighbor_index_mat: np.ndarray, nr_samples: int,
                                  n_neighbors: int, class_name1: str, class_name2: str) -> np.ndarray:
    """
    For each value of k up to n_neighbors, calculate the fraction of all kth neighbors that have the same
    class as the sample.
    """

    fraction_same_class = []

    max_n_neighbors = np.min([len(row) for row in neighbor_index_mat])
    max_neighbor_index = max_n_neighbors - 1
    n_neighbors = min(n_neighbors, max_neighbor_index)
    k_values = np.arange(1, n_neighbors + 1)
    for k in k_values:
        neighbor_k = [neighbor_index_mat[row][k] for row in range(len(neighbor_index_mat))] #for neighbor k of each sample
        class_neighbor1 = df_embeddings_with_meta.iloc[neighbor_k][class_name1].values
        class_neighbor2 = df_embeddings_with_meta.iloc[neighbor_k][class_name2].values
        fraction_same_class.append(np.mean(np.logical_and(
            class_neighbor1 == class1_sample,
            class_neighbor2 == class2_sample
        ))) #fraction of sample that have the same class
    fraction_same_class = np.array(fraction_same_class)
    return fraction_same_class, k_values

def evaluate_embeddings(dataset, meta_sel, knn_indices):

    biological_class_field, confounding_class_field = get_field_names_given_dataset(dataset)

    stats = calculate_fraction_same_class_per_k_value(meta_sel, knn_indices,
                                                      biological_class_field=biological_class_field,
                                                      confounding_class_field=confounding_class_field,
                                                      )

    return stats


def convert_types_in_stats(stats):
    for k, v in stats.items():
        if isinstance(v, np.ndarray):
            stats[k] = v.tolist()
        elif isinstance(v, (np.integer, np.int64)):
            stats[k] = int(v)
        elif isinstance(v, (np.floating, np.float64)):
            stats[k] = float(v)
        else:
            stats[k] = v
    return stats


def save_total_stats(stats, meta, dataset, model, results_folder, k_opt, bal_acc_at_k_opt):
    stats['k_opt'] = k_opt #store k_opt in stats
    stats['bal_acc_at_k_opt'] = bal_acc_at_k_opt #store max bal acc, obtained using k = k_opt
    index_k_opt = np.where(stats["k"] == k_opt)[0][0]  # index of k_opt in stats
    df_dict = {"k_opt": k_opt, "bal_acc_at_k_opt": bal_acc_at_k_opt}
    df_dict["robustness_index-k_opt"] = stats["robustness_index"][index_k_opt]
    df_dict["ID_performance-k_opt"] = stats["ID_performance"][index_k_opt]
    df_dict["OOD_performance-k_opt"] = stats["OOD_performance"][index_k_opt]
    df_dict["generalization_index-k_opt"] = stats["generalization_index"][index_k_opt]
    df_dict["SO_SS_ratio-k_opt"] = stats["SO_SS_ratio"][index_k_opt]

    if "robustness_index-mean" in stats and len(stats["robustness_index-mean"]) > index_k_opt:
        stats["robustness_index-mean-k_opt"] = stats["robustness_index-mean"][index_k_opt]
        stats["robustness_index-std-k_opt"] = stats["robustness_index-mean"][index_k_opt]
        df_dict["robustness_index-mean-k_opt"] = stats["robustness_index-mean-k_opt"]
        df_dict["robustness_index-std-k_opt"] = stats["robustness_index-std-k_opt"]

    biological_class_field, confounding_class_field = get_field_names_given_dataset(dataset)
    all_bio_classes = np.unique(meta[biological_class_field].values)
    all_conf_classes = np.unique(meta[confounding_class_field].values)

    fn = os.path.join(results_folder, f'{OutputFiles.FREQUENCIES}-{model}.pkl')
    with open(fn, 'wb') as f:
        pickle.dump({'stats': stats, 'all_bio_classes': all_bio_classes, 'all_conf_classes': all_conf_classes}, f)
    print(f'saved results to {fn}')

    # Store summary file
    output_file = os.path.join(results_folder, OutputFiles.SUMMARY_FILE)
    df_dict = convert_types_in_stats(df_dict)
    with open(output_file, 'w') as f:
        json.dump(df_dict, f, indent=4)

    print(f'saved results summary to {output_file}')
    return stats

def plot_3a_optimal_k_per_model(model, k_values_sel, bal_accs, k_opt, fig_folder):
    plt.plot(k_values_sel, bal_accs)
    plt.xlabel("k")
    plt.ylabel("Balanced accuracy")
    plt.title(f"Balanced accuracy vs k value for {model}\nOptimal k: {k_opt}")
    fn = os.path.join(fig_folder, f'3a-optimal-k.png')
    plt.savefig(fn, dpi=600)
    print(f"saved optimal k to {fn}")
    plt.close()


def plot_results_per_model(total_stats, k_values, model, fig_folder, dataset, bal_accs, k_opt):
    plot_3a_optimal_k_per_model(model, k_values, bal_accs, k_opt, fig_folder)
    plot_4a_freq_4_combinations_per_model(total_stats, fig_folder, dataset, model)
    plot_4b_freq_4_combinations_per_model_cum(total_stats, fig_folder, dataset, model)
    plot_4d_freq_2_combinations_per_model(total_stats, fig_folder, dataset, model)


def plot_4a_freq_4_combinations_per_model(stats, fig_folder, dataset, model):
    plt.figure()
    plt.plot(stats['fraction_SS-norm'], label='same bio class, same conf class')
    plt.plot(stats['fraction_SO-norm'], label='same bio class, other conf class')
    plt.plot(stats['fraction_OS-norm'], label='other bio class, same conf class')
    plt.plot(stats['fraction_OO-norm'], label='other bio class, other conf class')
    plt.legend()
    plt.ylim([0,1])
    plt.title(f"Frequency of same / other biological / confounding class\n{dataset} {model}")
    plt.savefig('/tmp/stats-all-frequencies.png')
    fn = os.path.join(fig_folder, f'4a-freq-4-combinations-knn-neighbor-k.png')
    plt.savefig(fn, dpi=600)
    print(f"saved frequencies 4 combinations plot to {fn}", flush=True)

    res = {f'fraction_{combi}-norm': stats[f'fraction_{combi}-norm'] for combi in ['SS', 'SO', 'OS', 'OO']}
    df = pd.DataFrame(res)
    fn = os.path.join(fig_folder, f'4a-freq-4-combinations-knn-neighbor-k.csv')
    df.to_csv(fn, index=False)
    print(f"saved frequencies 4 combinations df to {fn}", flush=True)

    plt.close()

def plot_4b_freq_4_combinations_per_model_cum(stats, fig_folder, dataset, model):
    plt.figure()
    k_range = np.arange(1, len(stats['fraction_SS-cum-norm']) + 1)
    plt.plot(stats['fraction_SS-cum-norm'], label='same bio class, same conf class')
    plt.plot(stats['fraction_SO-cum-norm'], label='same bio class, other conf class')
    plt.plot(stats['fraction_OS-cum-norm'], label='other bio class, same conf class')
    plt.plot(stats['fraction_OO-cum-norm'], label='other bio class, other conf class')
    plt.legend()
    plt.ylim([0,1])
    plt.title(f"Frequency of same / other biological / confounding class\n{dataset} {model}")
    plt.savefig('/tmp/stats-all-frequencies.png')
    fn = os.path.join(fig_folder, f'4b-freq-4-combinations-knn-neighbor-k-sum.png')
    plt.savefig(fn, dpi=600)
    print(f"saved frequencies 4 combinations plot to {fn}", flush=True)

    res = {f'fraction_{combi}-cum-norm': stats[f'fraction_{combi}-cum-norm'] for combi in ['SS', 'SO', 'OS', 'OO']}
    df = pd.DataFrame(res)
    fn = os.path.join(fig_folder, f'4b-freq-4-combinations-knn.csv')
    df.to_csv(fn, index=False)
    print(f"saved frequencies 4 combinations df to {fn}", flush=True)

    plt.close()

def plot_4d_freq_2_combinations_per_model(stats, fig_folder, dataset, model):
    fso=stats['fraction_SO-cum-norm']
    fos=stats['fraction_OS-cum-norm']
    total = fso+fos
    fso = fso / total
    fos = fos / total

    rob = stats['robustness_index']
    rob_std = stats['robustness_index-std']

    #get first 4 colors matplotlib
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    plt.figure()
    plt.plot(fso, label='same bio class, other conf class', color=colors[1])
    plt.plot(fos, label='other bio class, same conf class', color=colors[2])
    plt.plot(rob, label='robustness index', color=colors[0], linestyle='--', alpha=0.9)
    plt.axhline(y=0.5, color='gray', linestyle='--', label='0.5')
    plt.ylim([0,1])
    plt.legend()
    plt.title(f"Normalized frequency of same / other biological / confounding class\n{dataset} {model}")
    fn = os.path.join(fig_folder, f'4d-freq-2-combinations-knn.png')
    plt.savefig(fn, dpi=600)
    print(f"saved frequencies 2 combinations plot org to {fn}", flush=True)
    plt.close()

    #now add nice bands with std in same color
    plt.figure()
    plt.plot(fso, label='same bio class, other conf class', color=colors[1])
    plt.plot(fos, label='other bio class, same conf class', color=colors[2])
    plt.plot(rob, label='robustness index', color=colors[0], linestyle='--', alpha=0.9)
    plt.fill_between(range(len(rob)),
                        rob - rob_std,
                        rob + rob_std,
                        alpha=0.2, color=colors[0])

    plt.axhline(y=0.5, color='gray', linestyle='--', label='0.5')
    plt.ylim([0,1])
    plt.legend()
    plt.title(f"Normalized frequency of same / other biological / confounding class\n{dataset} {model}")
    fn = os.path.join(fig_folder, f'4e-freq-2-combinations-knn-neighbor-k-std.png')
    plt.savefig(fn, dpi=600)
    print(f"saved frequencies 2 combinations plot org with std to {fn}", flush=True)
    plt.close()

    res = {f'fraction_{combi}-cum-norm': stats[f'fraction_{combi}-cum-norm'] for combi in ['SS', 'SO', 'OS', 'OO']}
    res["robustness_index"] = stats['robustness_index']
    if len(stats['robustness_index-std']) == len(res["robustness_index"]):
        res["robustness_index-mean"] = stats['robustness_index-mean']
        res["robustness_index-std"] = stats['robustness_index-std']
    df = pd.DataFrame(res)
    fn = os.path.join(fig_folder, f'4e-freq-2-combinations-knn-neighbor-k-std.csv')
    df.to_csv(fn, index=False)
    print(f"saved frequencies 2 combinations with std df to {fn}", flush=True)



def calculate_per_class_prediction_stats(biological_class_field, confounding_class_field, classes, model, meta, aucs_per_class_list, k_opt, results_folder):
    auc_values = [aucs_per_class_list[k][k_opt] for k in range(len(aucs_per_class_list))]
    if np.sum([k is None for k in auc_values])>0:
        print(f"auc_values contains None values; probably a 2-class problem.")
        return
    auc_per_class_avg = np.vstack(auc_values).mean(axis=0)
    std_per_class_avg = np.vstack(auc_values).std(axis=0)
    df = pd.DataFrame({"bio_class": classes, "auc": auc_per_class_avg, "std": std_per_class_avg})
    conf_classes = np.unique(meta[confounding_class_field].values)
    for conf_class in conf_classes:
        nr_combi = []
        for row_index in range(len(df)):
            bio_class = df.bio_class.values[row_index]
            nr_combi.append(np.sum(np.logical_and(meta[biological_class_field] == bio_class, meta[confounding_class_field] == conf_class)))
        df[conf_class] = nr_combi
    fn = os.path.join(results_folder, f'auc-bio-{model}-per-class.csv')
    df.to_csv(fn, index=False)
    print(f'saved aucs per class to {fn}', flush=True)


def save_balanced_accuracies(model, accuracies_bio, k_values, results_folder):
    print(f"accuracies_bio nr rows: {len(accuracies_bio)} len {len(accuracies_bio[0])}")
    min_row_length = np.min([len(row) for row in accuracies_bio])
    print(f"min_row_length: {min_row_length} for model {model}, k_values {len(k_values)} max {np.max(k_values)}")
    if min_row_length > len(k_values):
        print(f"min_row_length {min_row_length} is larger than k_values length {len(k_values)}; reducing it to match it.")
    if len(k_values) > min_row_length:
        print(f"reducing k_values from {len(k_values)} to {min_row_length} to match accuracies_bio")
        k_values = k_values[:min_row_length]
    min_row_length = min(min_row_length, len(k_values))  # ensure k_values is not longer than accuracies_bio
    accuracies_bio = [row[:min_row_length] for row in accuracies_bio]
    bal_accs = np.mean(accuracies_bio, axis=0)
    stds = np.std(accuracies_bio, axis=0)
    bio_class_prediction_result = pd.DataFrame({"k": k_values, "bal_acc": bal_accs, "std": stds})
    fn = os.path.join(results_folder, f'{OutputFiles.BALANCED_ACCURACY}-{model}.csv')
    bio_class_prediction_result.to_csv(fn, index=False)
    print(f'saved bal_accs to {fn}', flush=True)

    index_opt_k = np.argmax(bal_accs)
    bal_acc_at_k_opt = bal_accs[index_opt_k]
    k_opt = k_values[index_opt_k]
    print(f"found optimal k for model {model}: k_opt = {k_opt}")
    return k_opt, bal_acc_at_k_opt, bio_class_prediction_result


def aggregate_per_combi(stats, bc):

    k_range = np.arange(1, stats[bc]["fraction_SS-cum"] .shape[0]+1)
    fraction_SS = stats[bc]["fraction_SS-cum"] / k_range
    fraction_SO = stats[bc]["fraction_SO-cum"] / k_range
    fraction_OS = stats[bc]["fraction_OS-cum"] / k_range
    fraction_OO = stats[bc]["fraction_OO-cum"] / k_range

    return fraction_SS, fraction_SO, fraction_OS, fraction_OO


def calculate_robustness_index_at_k_opt(models, results_folder, k_opt, options_subfolder):
    k_opt_bio_pred_model = {}
    rob_index_at_k_opt = {}
    for m, model in enumerate(models):
        fn = get_file_path(results_folder, model, options_subfolder, f'{OutputFiles.FREQUENCIES}-{model}.pkl')
        results = pickle.load(open(fn, 'rb'))
        stats = results['stats']
        k_range = np.array(stats['k'])
        robustness_index = stats["robustness_index"]

        if type(k_opt) is dict:
            k_opt_value = k_opt[model]
        else:
            k_opt_value = k_opt
        k_opt_value = min(k_opt_value, k_range[-1])
        if not np.max(k_range) == len(k_range):
            raise ValueError("assuming k_range is a consecutive range starting from 1") #determine the index of k_opt based on k_range below if this happens
        k_opt_bio_pred_model[model] = k_opt_value
        index_k_opt = np.where(k_range==k_opt_value)[0][0]
        rob_index_at_k_opt[model] = robustness_index[index_k_opt]
        if np.isnan(rob_index_at_k_opt[model]):
            rob_index_str = "NaN"
        else:
            rob_index_str = f"{rob_index_at_k_opt[model]:.3f}"
        print(f"model {model} k_opt_bio {k_opt_bio_pred_model} robustness index {rob_index_str} ")

    return k_opt_bio_pred_model, rob_index_at_k_opt


def get_robustness_index_k_range(model, results_folder, options_subfolder):
    stats = get_stats(model, results_folder, options_subfolder)
    return np.array(stats["k"]), stats["robustness_index"], stats["robustness_index-mean"], stats["robustness_index-std"]


def get_optimal_prediction_results_avg_all_datasets(datasets, models, options):
    opt_prediction_results = {}
    for m, model in enumerate(models):
        bal_acc_values_list = []
        k_values_list = []
        min_length = 1e6
        k_opts = []
        for dataset in datasets:
            options_ds = get_default_dataset_options(dataset, options)
            results_folder_ds = options_ds["results_folder"]
            fn = os.path.join(results_folder_ds, f'{OutputFiles.BALANCED_ACCURACY}-{model}.csv') #get bal_acc for biological classification
            if not os.path.isfile(fn):
                raise ValueError(f'missing bal_acc file {fn}')
            bal_accs_bio = pd.read_csv(fn)
            bal_acc_values = bal_accs_bio.bal_acc.values
            mis = np.isnan(bal_acc_values)
            bal_accs_bio = bal_accs_bio[~mis].reset_index(drop=True)
            bal_acc_values = bal_accs_bio.bal_acc.values
            index_k_opt = np.argmax(bal_acc_values)
            bal_acc_values_list.append(bal_acc_values)
            k_values = bal_accs_bio.k.values
            k_opt = k_values[index_k_opt]
            k_opts.append(k_opt)
            k_values_list.append(k_values)
            min_length = int(np.min([min_length, len(bal_acc_values)]))

        bal_acc_values_list = [bal_acc_values_list[k][:min_length] for k in range(len(bal_acc_values_list))]
        k_values_list = [k_values_list[k][:min_length] for k in range(len(k_values_list))]
        avg_bal_acc = np.vstack(bal_acc_values_list).mean(axis=0)
        k_values = k_values_list[0]
        index_max_bal_acc = np.argmax(avg_bal_acc)
        max_bal_acc = avg_bal_acc[index_max_bal_acc]
        k_opt = np.mean(k_opts)
        index_nearest_k_opt = np.argmin(np.abs(k_values - k_opt))
        k_opt = k_values[index_nearest_k_opt]  #find the nearest k in the k_values list
        opt_prediction_results[model] = {"bal_acc": avg_bal_acc, "k": k_values, "index_max_bal_acc": index_max_bal_acc, "max_bal_acc": max_bal_acc, "k_opt": k_opt}
    return opt_prediction_results


def get_optimal_prediction_results_per_dataset(datasets, models, options):
    opt_prediction_results = {}
    for m, model in enumerate(models):
        opt_prediction_results[model] = {}
        bal_acc_values_list = []
        k_values_list = []
        for dataset in datasets:
            options_ds = get_default_dataset_options(dataset, options)
            results_folder_ds = options_ds["results_folder"]
            fn = os.path.join(results_folder_ds, f'{OutputFiles.BALANCED_ACCURACY}-{model}.csv') #get bal_acc for biological classification
            if not os.path.isfile(fn):
                raise ValueError(f'missing bal_acc file {fn}')
            bal_accs_bio = pd.read_csv(fn)
            bal_acc_values = bal_accs_bio.bal_acc.values
            mis = np.isnan(bal_acc_values)
            bal_accs_bio = bal_accs_bio[~mis].reset_index(drop=True)
            bal_acc_values = bal_accs_bio.bal_acc.values
            bal_acc_values_list.append(bal_acc_values)
            k_values = bal_accs_bio.k.values
            k_values_list.append(k_values)
            opt_index = np.argmax(bal_acc_values)
            max_bal_acc = bal_acc_values[opt_index]
            k_opt = k_values[opt_index]
            opt_prediction_results[model][dataset] = {"max_bal_acc": max_bal_acc, "k_opt": k_opt}
    return opt_prediction_results


def get_robustness_results_median_k_opt_per_dataset(datasets, models, options):
    max_k = 599
    model_robustness_index = {}
    model_avg_k_opt = {}
    model_robustness_index_at_k_opt = {}
    robustness_index_vectors = {}
    dataset_median_k_opt = {}
    ok = True
    k_ranges = {}
    for dataset in datasets:
        k_ranges[dataset] = []
        dataset_k_opts = []
        for model in models:
            if not model in robustness_index_vectors:
                robustness_index_vectors[model] = {}
            options_tmp = get_default_dataset_options(dataset, options)
            results_folder, fig_folder = get_folder_paths(options_tmp, dataset, model)
            stats = get_stats(model, results_folder)
            k_range, robustness_index, _, robustness_index_std = get_robustness_index_k_range(model, results_folder)
            k_opt = int(stats['k_opt'])
            dataset_k_opts.append(k_opt)

            if k_range is None:
                print(f"skipping model {model} for dataset {dataset}")
                ok = False
                break
            k_range = k_range[:max_k]
            robustness_index = robustness_index[:max_k]
            k_ranges[dataset].append(k_range)
            robustness_index_vectors[model][dataset] = robustness_index
        dataset_median_k_opt[dataset] = np.median(dataset_k_opts)

    if ok:
        mean_median_opt_k = int(np.mean(list(dataset_median_k_opt.values())))
        for model in models:
            k_range = np.array(k_ranges[datasets[0]][0])
            robustness_index_vectors_model = list(robustness_index_vectors[model].values())
            min_length = np.min([len(rv) for rv in robustness_index_vectors_model])
            robustness_index_vectors_red = [rv[:min_length] for rv in robustness_index_vectors_model]
            k_range = k_range[:min_length]
            max_k = min(max_k, min_length)
            avg_robustness_index = np.vstack(robustness_index_vectors_red).mean(axis=0)
            model_robustness_index[model] = avg_robustness_index
            k_diff = np.abs(k_range-mean_median_opt_k)
            index_closest_k_opt = int(np.argmin(k_diff))
            model_avg_k_opt[model] = k_range[index_closest_k_opt]
            model_robustness_index_at_k_opt[model] = float(avg_robustness_index[index_closest_k_opt])

    return model_robustness_index, np.array(k_range), max_k, model_avg_k_opt, model_robustness_index_at_k_opt


def get_robustness_results_all_datasets(datasets, models, options):
    max_k = 599
    model_robustness_index = {}
    model_robustness_index_at_k_opt = {}

    k_opts_per_dataset = get_k_opt_per_dataset(datasets, models, options)
    avg_median_k_opt_all_datasets = int(round(np.mean(list(k_opts_per_dataset.values()))))
    print(f"average median k_opt across all datasets: {avg_median_k_opt_all_datasets}")

    for model in models:
        k_ranges = []
        robustness_index_vectors = []
        ok = True
        k_opt_values = []
        for dataset in datasets:
            options_tmp = get_default_dataset_options(dataset, options)
            results_folder, fig_folder = get_folder_paths(options_tmp, dataset, model)
            k_range, robustness_index, _, robustness_index_std = get_robustness_index_k_range(model, results_folder)

            if k_range is None:
                print(f"skipping model {model} for dataset {dataset}")
                ok = False
                break
            k_range = k_range[:max_k]
            robustness_index = robustness_index[:max_k]
            k_ranges.append(k_range)
            robustness_index_vectors.append(robustness_index)

        if ok:
            k_range = k_ranges[0]
            min_length = np.min([len(rv) for rv in robustness_index_vectors])
            robustness_index_vectors = [rv[:min_length] for rv in robustness_index_vectors]
            k_range = k_range[:min_length]
            max_k = min(max_k, min_length)
            avg_robustness_index = np.vstack(robustness_index_vectors).mean(axis=0)
            model_robustness_index[model] = avg_robustness_index
            k_diff = np.abs(k_range-avg_median_k_opt_all_datasets)
            index_closest_k_opt = int(np.argmin(k_diff))
            model_robustness_index_at_k_opt[model] = float(avg_robustness_index[index_closest_k_opt])

    return model_robustness_index, np.array(k_range), max_k, avg_median_k_opt_all_datasets, model_robustness_index_at_k_opt


def get_robustness_results_per_dataset(datasets, models, options):
    model_robustness_index = {}
    for model in models:
        model_robustness_index[model] = {}
        for dataset in datasets:
            options_tmp = get_default_dataset_options(dataset, options)
            results_folder, fig_folder = get_folder_paths(options_tmp, dataset, model)
            k_range, robustness_index, _, robustness_index_std = get_robustness_index_k_range(model, results_folder)
            if k_range is None:
                print(f"skipping model {model} for dataset {dataset}")
                break
            model_robustness_index[model][dataset] = {"k_range": k_range, "robustness_index": robustness_index}
    return model_robustness_index


def get_model_colors(models):
    return [[float(k) for k in plt.get_cmap("tab20")(i)] for i in range(len(models))]


def get_stats(model, results_folder, options_subfolder):
    fn = get_file_path(results_folder, model, options_subfolder, f'{OutputFiles.FREQUENCIES}-{model}.pkl')
    # if not os.path.exists(fn):
    #     print(f"file not found: {fn}")
    #     return None, None, None
    results = pickle.load(open(fn, 'rb'))
    stats = results['stats']
    return stats


def get_default_dataset_options(dataset, options):
    options_tmp = copy.deepcopy(options)
    max_patches_per_combi_model = {"tcga-uniform-subset": -1, "tcga-2k": -1, "tcga-4x4": -1, "camelyon16": -1, "camelyon17": -1, "tolkach-esca": -1}
    options_tmp["max_patches_per_combi"] = max_patches_per_combi_model[dataset]
    return options_tmp


def get_k_opt_per_dataset(datasets, models, options):
    k_opts_per_dataset = {}
    for dataset in datasets:
        k_opts_per_dataset[dataset] = {}
        options_tmp = get_default_dataset_options(dataset, options)
        k_opts = []
        for model in models:
            results_folder, fig_folder = get_folder_paths(options_tmp, dataset, model)
            stats = get_stats(model, results_folder)
            if stats is not None:
                k_opt = int(stats['k_opt'])
                k_opts.append(k_opt)
            else:
                print(f"skipping model {model} for dataset {dataset} due to missing stats")
        dataset_k_opt = int(np.median(k_opts))
        print(f"dataset {dataset} median k_opt {dataset_k_opt}")
        k_opts_per_dataset[dataset] = dataset_k_opt
    return k_opts_per_dataset


def get_folder_paths(options, dataset, model):
    results_folder = Path(options["results_dir"])
    fig_subfolder = Path(options["figures_subdir"])
    max_patches_per_combi = options["max_patches_per_combi"]
    k_opt_param = options["k_opt_param"]

    args_subfolder = f"{max_patches_per_combi}_{k_opt_param}"

    if options["DBG"]:
        results_folder = results_folder / "debug"

    results_folder = results_folder / model / dataset / args_subfolder
    fig_folder = results_folder / fig_subfolder    # TODO: are they by model or generic?

    print(f"using results_folder: {results_folder}, fig_folder: {fig_folder}")

    options["results_folder"] = results_folder
    options["fig_folder"] = fig_folder

    results_folder.mkdir(parents=True, exist_ok=True)
    fig_folder.mkdir(parents=True, exist_ok=True)

    return results_folder, fig_folder


def get_generic_folder_paths(options, dataset):
    results_folder = Path(options["results_dir"])
    fig_subfolder = Path(options["figures_subdir"])
    max_patches_per_combi = options["max_patches_per_combi"]
    k_opt_param = options["k_opt_param"]

    args_subfolder = f"{max_patches_per_combi}_{k_opt_param}"

    fig_folder = results_folder / fig_subfolder
    options_subfolder = Path(dataset) / args_subfolder

    print(f"using results_folder: {results_folder}, fig_folder: {fig_folder}")

    options["results_folder"] = results_folder
    options["fig_folder"] = fig_folder

    results_folder.mkdir(parents=True, exist_ok=True)
    fig_folder.mkdir(parents=True, exist_ok=True)

    return results_folder, fig_folder, options_subfolder


def get_model_names(base_folder: str):
    base_folder = Path(base_folder)
    model_folders = [item.stem for item in base_folder.iterdir() if item.is_dir() and item.name != "fig"]
    return model_folders


def get_file_path(results_path, model, options_subfolder, output_file):
    file_path = results_path / model / options_subfolder / output_file
    if not file_path.exists():
        raise ValueError(f'missing file {file_path}')
    return file_path


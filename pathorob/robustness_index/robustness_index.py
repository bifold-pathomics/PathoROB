import os
import time
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, Normalizer

from pathorob.features.data_manager import FeatureDataManager
from pathorob.features.constants import AVAILABLE_DATASETS
import pathorob.robustness_index.robustness_graphs as robustness_graphs
from pathorob.robustness_index.robustness_index_paired import evaluate_model_pairs
from pathorob.robustness_index.robustness_index_utils import (
    aggregate_stats, save_total_stats, get_field_names_given_dataset, evaluate_knn_accuracy,
    get_k_values,
    save_balanced_accuracies, evaluate_embeddings, calculate_per_class_prediction_stats,
    calculate_robustness_index_at_k_opt, get_model_colors, get_folder_paths, plot_results_per_model,
    OutputFiles, get_model_names, get_generic_folder_paths, get_file_path
)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser(description='Calculate robustness index for a given dataset and model.')

    parser.add_argument('--mode', type=str, choices=['compute', 'compare'], default='compute',
                        help='Mode to run: "compute" to calculate robustness index for a single model, '
                             '"compare" to compare multiple models, requires robustness index computed for all models.')

    #required parameters
    parser.add_argument('--model', type=str, help='Model name.')
    parser.add_argument(
        "--dataset", type=str, nargs="+", default=AVAILABLE_DATASETS,
        help=f"PathoROB datasets on which the robustness index is computed. Available datasets: {AVAILABLE_DATASETS}."
    )
    #optional parameters
    parser.add_argument('--features_dir', type=str, default="data/features", help='Folder for embeddings. The features should be stored in this folder: [features_dir]/[model]/[dataset].')
    parser.add_argument('--metadata_dir', type=str, default="data/metadata", help='Folder for metadata.')
    parser.add_argument('--results_dir', type=str, default="results/robustness_index", help='Root folder for results.')
    parser.add_argument('--figures_subdir', type=str, default="fig", help='Root folder for figures.')
    parser.add_argument('--paired_evaluation', type=str2bool, default=None, help='Whether to use paired evaluation. Per default (None), this is True for tcga and False for the other datasets.')
    parser.add_argument('--k_opt_param', type=int, default=0, help='This parameter can be set to a specific k value to report the robustness index for the specified value of k. '
                                                                      'By default, 0 is used, which means the default values per dataset are used. '
                                                                      'If -1, results are produced for all values of k, the optimal k value will be optimized based on biological class prediction.')
    parser.add_argument('--max_patches_per_combi', type=int, default=-1, help='Maximum patches per combination. -1 for no limit, or a specific number to limit the dataset size.')
    parser.add_argument('--compute_bootstrapped_robustness_index', action='store_true', help='Compute bootstrapped robustness index.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for parallel processing.')
    parser.add_argument('--plot_graphs', action='store_true', help='Plot graphs when flag is provided.')
    parser.add_argument('--plots_wo_legend', action='store_true', help='Plot graphs without legend when flag is provided. Only some plots accept this option')
    parser.add_argument('--debug_mode', action='store_true', help='Run in debug mode with limited data for faster testing.')

    return parser.parse_args()


def check_sufficient_unique_values(X_train, X_test, y_train, y_test, model, results_folder):
    nr_unique_values = len(np.unique(X_train.flatten()))
    nr_zero_embeddings = np.sum(X_train.sum(axis=1) == 0)
    if nr_zero_embeddings > 0:
        raise ValueError(f'X_train contains {nr_zero_embeddings} zero embeddings')
    if nr_unique_values == 1:
        raise ValueError("X_train only has 1 value")

    nr_unique_values = len(np.unique(X_test.flatten()))
    if nr_unique_values == 1:
        raise ValueError("X_test only has 1 value")

    nr_unique_values = len(np.unique(y_train.flatten()))
    if nr_unique_values == 1:
        raise ValueError("y_train only has 1 value")

    nr_unique_values = len(np.unique(y_test.flatten()))
    if nr_unique_values == 1:
        raise ValueError("y_test only has 1 value")


def select_optimal_k_value(dataset, model, patch_names, embeddings, meta, results_folder, fig_folder,
                           num_workers = 8, compute_bootstrapped_robustness_index=False, do_checks=False, opt_k=None, plot_graphs=True):
    #perform train/val split at the confounding-class level (medical center level) to prevent biases and measure OOD performance
    biological_class_field, confounding_class_field = get_field_names_given_dataset(dataset)

    max_samples_per_group = int(np.max(meta["slide_id"].value_counts().values))

    k_values = get_k_values(dataset, False, opt_k, max_samples_per_group)

    bio_values = meta[biological_class_field].values
    bio_classes = np.unique(bio_values)
    print(f"dataset {dataset} nr bio_classes {len(bio_classes)}", flush=True)

    accuracies_bio = []
    aucs_per_class_list = []
    all_stats = []

    t_rep_start = time.time()

    X_scaled = embeddings

    if len(np.unique(X_scaled.flatten())) == 1:
        raise ValueError("X_scaled only has 1 value")

    X_train = X_scaled #the test case itself will be removed from its set of training neighbors
    X_test = X_scaled
    nr_samples_train = len(X_train)
    nr_samples_test = len(X_test)
    nr_samples = min(nr_samples_train, nr_samples_test)
    print(f"knn dataset size X_train {X_train.shape} X_test {X_test.shape}", flush=True)

    print("dbg encode labels", flush=True)
    y_train = bio_values
    y_test = bio_values

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    nr_train_bio_classes = len(np.unique(y_train))
    nr_test_bio_classes = len(np.unique(y_test))

    print(f'nr unique target bio_classes: train {nr_train_bio_classes} test {nr_test_bio_classes}', flush=True)
    if not (nr_train_bio_classes == nr_test_bio_classes):
        raise ValueError(f"train and test bio_classes do not match: {nr_train_bio_classes} {nr_test_bio_classes}")

    if do_checks:
        check_sufficient_unique_values(X_train, X_test, y_train, y_test, model, results_folder)
        if np.any(np.linalg.norm(X_train, axis=1) == 0) or np.any(np.linalg.norm(X_test, axis=1) == 0):
            raise ValueError(
                "X_train or X_test contains zero vectors, which can cause division by zero in distance calculation.")

    accuracies_k_bio = []
    k_values_sel = [k for k in k_values if k <= nr_samples]
    aucs_per_class = {}
    knn_distances, knn_indices = None, None
    acc_score_bio, auc_per_class, knn_distances, knn_indices, effective_n_neighbors = evaluate_knn_accuracy(meta, dataset, X_train, X_test,
                                                                                     y_train, y_test, k_values_sel[-1], num_workers,
                                                                                     knn_distances, knn_indices)
    max_k = knn_indices.shape[1]
    k_values_sel = [k for k in k_values_sel if k <= max_k]
    if opt_k and opt_k > 0:
        acc_score_bio, auc_per_class, knn_distances, knn_indices, effective_n_neighbors = evaluate_knn_accuracy(meta, dataset, X_train, X_test,
                                                                                         y_train, y_test, opt_k,
                                                                                         num_workers, knn_distances,
                                                                                         knn_indices)
        acc_score_bio = float(acc_score_bio)
        accuracies_k_bio.append(acc_score_bio)
        aucs_per_class[opt_k] = auc_per_class
    else:
        for k in k_values_sel[::-1]: #iterate over k values in reverse order to find the optimal k value
            t0 = time.time()
            acc_score_bio, auc_per_class, knn_distances, knn_indices, effective_n_neighbors = evaluate_knn_accuracy(meta, dataset, X_train, X_test, y_train, y_test, k, num_workers, knn_distances, knn_indices)
            acc_score_bio = float(acc_score_bio)
            accuracies_k_bio.append(acc_score_bio)
            aucs_per_class[k] = auc_per_class
            dt = time.time() - t0
            print(f"select_optimal_k_value k {k} dt {dt:.2f}", flush=True)
        accuracies_k_bio = accuracies_k_bio[::-1]  # reverse back again to match the regular k_values_sel order

        # TODO: error here when using fixed k
        index_max_bal_acc = np.argmax(accuracies_k_bio)
        opt_k = k_values_sel[index_max_bal_acc]

    print(f"opt-k {opt_k} max balanced accuracy {np.max(accuracies_k_bio):.4f} accuracies_k_bio {[f'{float(f):.3f}' for f in accuracies_k_bio]}", flush=True)
    accuracies_bio.append(accuracies_k_bio)
    aucs_per_class_list.append(aucs_per_class)
    t_rep_end = time.time()
    print(f"dt {t_rep_end - t_rep_start:.2f} sec", flush=True)
    stats_rep = evaluate_embeddings(dataset, meta, knn_indices)
    all_stats.append(stats_rep)

    total_stats = aggregate_stats(all_stats, compute_bootstrapped_robustness_index=compute_bootstrapped_robustness_index)
    max_k = total_stats["max_k"]
    k_values_sel = [k for k in k_values_sel if k <= max_k]
    nr_k_sel = len(k_values_sel)
    accuracies_bio = [row[:nr_k_sel] for row in accuracies_bio]

    k_values = np.array([k for k in k_values if k <= max_k])
    k_opt, bal_acc_at_k_opt, bio_class_prediction_result = save_balanced_accuracies(model, accuracies_bio, k_values, results_folder)
    total_stats = save_total_stats(total_stats, meta, dataset, model, results_folder, k_opt, bal_acc_at_k_opt)
    calculate_per_class_prediction_stats(biological_class_field, confounding_class_field, bio_classes, model, meta, aucs_per_class_list, k_opt, results_folder)
    if plot_graphs:
        plot_results_per_model(total_stats, k_values, model, fig_folder, dataset,
                               bio_class_prediction_result["bal_acc"], k_opt)
    return k_opt, bio_class_prediction_result, total_stats


def evaluate_model(
        dataset, data_manager, model, meta, results_folder, fig_folder, num_workers=8, k_opt_param=-1, compute_bootstrapped_robustness_index=False, DBG=False, plot_graphs=True):
    embeddings = data_manager.load_features(model, dataset, meta)
    print('loaded all embeddings')

    print(f"embeddings before normalize shape {embeddings.shape} max {np.max(embeddings)} min {np.min(embeddings)}")
    normalizer = Normalizer(norm='l2')
    embeddings = normalizer.fit_transform(embeddings)
    print(f"embeddings after normalize shape {embeddings.shape} max {np.max(embeddings)} min {np.min(embeddings)}")

    patch_names = np.array(meta["patch_name"].values)

    print(f"len meta {len(meta)} patch_names {len(patch_names)} emb {len(embeddings)}")
    print("len index before ",len(meta.index),"unique",len(np.unique(meta.index)))

    k_opt, bio_class_prediction_results, robustness_metrics_dict = select_optimal_k_value(
        dataset, model, patch_names, embeddings, meta, results_folder, fig_folder,
        num_workers=num_workers, compute_bootstrapped_robustness_index=compute_bootstrapped_robustness_index,
        opt_k=k_opt_param, plot_graphs=plot_graphs
    )
    print(f"found k_opt {k_opt}")
    return bio_class_prediction_results, robustness_metrics_dict


def calc_rob_index_model(paired_eval, data_manager, model, dataset, meta, results_folder, fig_folder, num_workers=8, k_opt_param=-1, compute_bootstrapped_robustness_index=False, DBG=False, plot_graphs=True):
    results = {}
    robustness_metrics_dict = {}

    if paired_eval:
        bio_class_prediction_results, robustness_metrics = evaluate_model_pairs(dataset, data_manager, model, meta, results_folder, fig_folder, num_workers=num_workers, k_opt_param=k_opt_param, DBG=DBG, compute_bootstrapped_robustness_index=compute_bootstrapped_robustness_index, plot_graphs=plot_graphs)
    else:
        bio_class_prediction_results, robustness_metrics = evaluate_model(dataset, data_manager, model, meta, results_folder, fig_folder, num_workers=num_workers, k_opt_param=k_opt_param,  compute_bootstrapped_robustness_index=compute_bootstrapped_robustness_index, DBG=DBG, plot_graphs=plot_graphs)
    results[model] = bio_class_prediction_results
    robustness_metrics_dict[model] = robustness_metrics
    return results, robustness_metrics_dict


def get_meta(data_manager, dataset, paired_evaluation):
    if dataset == "tcga":
        if paired_evaluation:
            metadata_name = "tcga_2x2"
        else:
            metadata_name = "tcga_4x4"
    elif dataset == "camelyon":
        if paired_evaluation:
            raise ValueError(f"Paired evaluation not implemented for dataset: {dataset}")
        else:
            metadata_name = "camelyon"
    elif dataset == "tolkach_esca":
        if paired_evaluation:
            raise ValueError(f"Paired evaluation not implemented for dataset: {dataset}")
        else:
            metadata_name = "tolkach_esca_reduced"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    meta = data_manager.load_metadata(metadata_name)
    meta.insert(0, "patch_name", data_manager.compute_ids(meta))
    # Exclude OOD data if present in the metadata frame
    meta = meta[~(meta["subset"] == "OOD")].reset_index(drop=True)
    return meta


def get_bal_acc_values(results_folder, model, options_subfolder,):
    fn = get_file_path(results_folder, model, options_subfolder, OutputFiles.BALANCED_ACCURACIES)
    bal_accs_bio = pd.read_csv(fn)
    bal_acc_values = bal_accs_bio.bal_acc.values
    mis = np.isnan(bal_acc_values)
    bal_accs_bio = bal_accs_bio[~mis].reset_index(drop=True)
    bal_acc_values = bal_accs_bio.bal_acc.values
    k_values = bal_accs_bio.k.values
    index_opt = np.argmax(bal_acc_values)
    k_opt = k_values[index_opt]
    return bal_acc_values, k_opt


def report_optimal_k(results_folder, fig_folder, models, plots_wo_legend, options_subfolder):
    print("Optimal k values")

    plt.figure(figsize=(5, 4))
    opt_k_values = []
    mcolors = get_model_colors(models)
    model_k_opt = {}

    max_bal_acc_values = []
    max_bal_acc_value = {}
    for m, model in enumerate(models):
        fn = get_file_path(results_folder, model, options_subfolder, OutputFiles.BALANCED_ACCURACIES)
        bal_accs_bio = pd.read_csv(fn)
        bal_acc_values = bal_accs_bio.bal_acc.values
        mis = np.isnan(bal_acc_values)
        bal_accs_bio = bal_accs_bio[~mis].reset_index(drop=True)
        bal_acc_values = bal_accs_bio.bal_acc.values
        max_bal_acc = np.max(bal_acc_values)
        max_bal_acc_values.append(max_bal_acc)
        max_bal_acc_value[model] = max_bal_acc

    sortindex = np.argsort(max_bal_acc_values)
    sortindex = sortindex[::-1]

    #generate legend
    for m in range(len(models)):
        index = sortindex[m]
        model = models[index]
        bal_acc_values, k_opt = get_bal_acc_values(results_folder, model, options_subfolder)
        index_opt = np.argmax(bal_acc_values)
        max_bal_acc = bal_acc_values[index_opt]

        opt_k_values.append(k_opt)
        model_k_opt[model] = k_opt
        print(f"model {model} optimal k: {k_opt}")
        plt.plot(k_opt, bal_acc_values[index_opt], 'o', color = mcolors[m], label=f"{model} k={k_opt} {max_bal_acc:.3f}")

    sortindex = sortindex[::-1]

    model_bal_acc_values = {}

    #plot top lines last
    for m in range(len(models)):
            index = sortindex[m]
            model = models[index]
            fn = get_file_path(results_folder, model, options_subfolder, OutputFiles.BALANCED_ACCURACIES)
            bal_accs_bio = pd.read_csv(fn)
            mis = np.isnan(bal_accs_bio.bal_acc.values)
            bal_accs_bio = bal_accs_bio[~mis]
            bal_acc_values = bal_accs_bio.bal_acc.values
            k_values = bal_accs_bio.k.values

            index_opt = np.argmax(bal_acc_values)
            k_opt = k_values[index_opt]
            model_k_opt[model] = k_opt
            print(f"model {model} optimal k: {k_opt}")

            df = pd.DataFrame({'k': k_values, 'bal_acc': bal_acc_values})
            model_bal_acc_values[model] = df
            plt.plot(k_values, bal_acc_values, color=mcolors[m])
            plt.plot(k_opt, bal_acc_values[index_opt], 'o', color=mcolors[m])


    if len(models) > 1:
        model_str = "all-models"
    else:
        model_str = models[0]

    plt.xlabel("k")
    plt.ylabel("Balanced accuracy")
    plt.title(f"Optimal k values")
    if not plots_wo_legend:
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
        plt.gcf().set_size_inches(12, 6)
    plt.tight_layout()
    fn = fig_folder / f'optimal-k-values-{model_str}.png'
    plt.savefig(fn, dpi=600)
    print(f"saved optimal k values to {fn}")
    return model_k_opt, model_bal_acc_values, max_bal_acc_value


def reduce_dataset(meta, max_patches_per_combi):
    np.random.seed(123)  # always use same fixed seed on purpose here for reproducibility
    meta = meta.sample(frac=1).reset_index(drop=True)
    meta["bio_conf_combi"] = meta["biological_class"] + "-" + meta["medical_center"]
    if max_patches_per_combi > 0:
        nr_org = len(meta)
        meta = meta.groupby(["subset", "bio_conf_combi"]).head(max_patches_per_combi).reset_index(drop=True)
        print(f"reduced dataset to {max_patches_per_combi} patches per combination from {nr_org} to {len(meta)}")
    return meta


def results_summary(model, meta, max_patches_per_combi, results_folder, model_k_opt, median_k_opt, results, dt):
    nr_patches = len(meta)
    result = {}

    print(f"results_summary {model}")

    model_bal_acc = results[model]["bal_acc_at_k_opt"]
    bal_acc = np.max(model_bal_acc)
    model_k_opt = results[model]["k_opt"]
    index_k_opt_full = model_k_opt - 1 #index in full range 1..k_max

    rob_index_k_opt = results[model]['robustness_index'][index_k_opt_full]
    model_rob_index_mean = results[model]['robustness_index-mean']
    model_rob_index_std = results[model]['robustness_index-std']
    print(f"model_k_opt {model_k_opt}  median_k_opt {median_k_opt}")
    bootstrapping_avail = len(model_rob_index_mean) > 1

    if bootstrapping_avail:
        mean_std_str = f"robustness_index mean {model_rob_index_mean[index_k_opt_full]:.3f}  std {model_rob_index_std[index_k_opt_full]:.3f}"
        mean_std_str_median = f"robustness_index mean {model_rob_index_mean[index_median_k_opt]:.3f} std {model_rob_index_std[index_median_k_opt]:.3f}"
    else:
        mean_std_str = f"robustness_index mean -1 std -1"
        mean_std_str_median = "robustness_index mean -1 std -1"

    result_string = (f"final result max-patches-per-combi {max_patches_per_combi} "
                     f"model {model} nr-patches {nr_patches} "
                     f"model k_opt {model_k_opt} robustness_index {rob_index_k_opt :.3f} {mean_std_str} "
                     f"bal_acc {bal_acc:.3f} {model_bal_acc:.3f} "                                 
                     f" runtime {dt:.2f} sec, {dt/60.0:.2f} min")

    model_result = {"model": model, "median_k_opt": median_k_opt, "balanced_accuracy": bal_acc}
    result[model] = model_result

    print(result_string)
    return result


def plot_all_results(models, results_folder, fig_folder, model_k_opt, median_k_opt, dataset, options, plots_wo_legend, options_subfolder):
    boostrapped_robustness_index = options.get("compute_bootstrapped_robustness_index", False)

    robustness_graphs.plot11_performance_robustness_tradeoff(models, options, results_folder, fig_folder, model_k_opt, median_k_opt, dataset, options_subfolder)

    robustness_graphs.plot_4_freq_bio_vs_conf_all_models(models, results_folder, fig_folder, plots_wo_legend, options_subfolder)
    robustness_graphs.plot_5_freq_bio_vs_conf_all_models(models, results_folder, fig_folder, plots_wo_legend, options_subfolder)
    _                     , _                = robustness_graphs.plot_6_robustness_index_all_models(models, results_folder, fig_folder, model_k_opt, median_k_opt, True, dataset, boostrapped_robustness_index, plots_wo_legend, options_subfolder)
    robustness_metrics, robustness_index = robustness_graphs.plot_6_robustness_index_all_models(models, results_folder, fig_folder, model_k_opt, median_k_opt, False, dataset,  boostrapped_robustness_index, plots_wo_legend,  options_subfolder) #return this as default below

    plot_all_dataset_results = True
    if plot_all_dataset_results:
        datasets = ["camelyon", "tcga", "tolkach_esca"]
        try:
            robustness_graphs.plot_8_robustness_index_all_datasets(datasets, models, options) #use median k_opt per dataset
            robustness_graphs.plot_10_pareto_plot_avg_all_datasets(datasets, models, options)
            #plot_10a_robustness_graphs.pareto_plot_all_datasets(datasets, models, options)
        except Exception as e:
            print(f"exception plotting all-dataset graphs (input from other runs may not yet be available): {e}")
            pass

    return robustness_metrics, robustness_index


def get_median_k_opt_given_dataset(dataset):
    if dataset == "tcga":
        median_k_opt = 61
    elif dataset == "camelyon":
        median_k_opt = 11
    elif dataset == "tolkach_esca":
        median_k_opt = 46
    else:
        raise ValueError(f"unknown dataset {dataset} for median k_opt")
    return int(median_k_opt)


def compute(
        model: str,
        dataset: str,
        features_dir: str = "data/features",
        metadata_dir: str = "data/metadata",
        results_dir: str = "results/robustness_index",
        figures_subdir: str = "results/robustness_index/fig",
        paired_evaluation: bool = None,
        k_opt_param: int = 0,
        max_patches_per_combi: int = -1,
        compute_bootstrapped_robustness_index: bool = False,
        num_workers: int = 8,
        plot_graphs: bool = False,
        plots_wo_legend: bool = False,
        debug_mode: bool = False,
):
    t_start = time.time()

    if paired_evaluation is None:
        # default: use paired setup for TCGA, as it has many biological and confounding classes and is not balanced
        paired_evaluation = dataset == "tcga"

    # Use median k value if not specified
    if k_opt_param == 0:
        k_opt_param = get_median_k_opt_given_dataset(dataset)

    options = {
        "model": model,
        "max_patches_per_combi": max_patches_per_combi,
        "k_opt_param": k_opt_param,
        "dataset": dataset,
        "results_dir": results_dir,
        "figures_subdir": figures_subdir,
        "paired_evaluation": paired_evaluation,
        "metadata_dir": metadata_dir
    }

    print("using these settings:")
    for param in options:
        print(f"param {param}: {options[param]}")

    DBG=debug_mode
    options["DBG"] = DBG
    results_folder, fig_folder = get_folder_paths(options, dataset, model)

    data_manager = FeatureDataManager(features_dir=features_dir, metadata_dir=metadata_dir)
    meta = get_meta(data_manager, dataset, options["paired_evaluation"])
    meta = reduce_dataset(meta, max_patches_per_combi=max_patches_per_combi)

    robustness_metrics_dict, results = calc_rob_index_model(options["paired_evaluation"], data_manager, model, dataset, meta,
                                                      results_folder, fig_folder,
                                                      num_workers=num_workers,
                                                      k_opt_param=k_opt_param,
                                                      compute_bootstrapped_robustness_index=compute_bootstrapped_robustness_index,
                                                      DBG=DBG, plot_graphs=plot_graphs)

    t_end_calc = time.time()
    dt = t_end_calc - t_start
    print(f"calculation time {dt:.2f} seconds = {dt/60:.2f} minutes = {dt/3600:.2f} hours")

    k_opt = results[model]['k_opt'] if k_opt_param == -1 else k_opt_param
    if plot_graphs:
        robustness_graphs.plot_results(model, results_folder, fig_folder, k_opt)

    if results:
        median_k_opt = get_median_k_opt_given_dataset(dataset)
        result = results_summary(model, meta, max_patches_per_combi, results_folder, k_opt, median_k_opt, results, dt)
        print("final result", result)

    return robustness_metrics_dict


def compare(
        dataset: str,
        results_dir: str = "results/robustness_index",
        figures_subdir: str = "results/robustness_index/fig",
        k_opt_param: int = -1,
        max_patches_per_combi: int = -1,
        plots_wo_legend: bool = False,
        compute_bootstrapped_robustness_index: bool = False,
        **kwargs
):
    options = {
        "max_patches_per_combi": max_patches_per_combi,
        "k_opt_param": k_opt_param,
        "results_dir": results_dir,
        "figures_subdir": figures_subdir,
    }

    models = get_model_names(results_dir)
    results_folder, fig_folder, options_subfolder = get_generic_folder_paths(options, dataset)

    if k_opt_param == -1:
        model_k_opt, model_bal_acc_values, max_bal_acc_value = report_optimal_k(results_folder, fig_folder, models, plots_wo_legend, options_subfolder)
        median_k_opt = get_median_k_opt_given_dataset(dataset)
        # print(f"dataset {dataset} found model k_opt {model_k_opt}  median k_opt: {median_k_opt:.2f}")
    else: #fixed k_opt_param
        print(f"using fixed k_opt_param {k_opt_param}")
        model_k_opt = {model: k_opt_param for model in models}
        median_k_opt = k_opt_param
        model_bal_acc_values=None

    robustness_metrics, robustness_index = plot_all_results(models, results_folder, fig_folder, model_k_opt, median_k_opt,
                                                            dataset, options, plots_wo_legend, options_subfolder)

    robustness_graphs.pareto_plot(dataset, models, model_bal_acc_values, robustness_metrics, fig_folder)


def compute_all(args_dict):
    datasets = args_dict.pop('dataset')
    print(f"Start robustness index calculation for model {args_dict['model']} on datasets: {datasets}.")
    for dataset in datasets:
        compute(**args_dict, dataset=dataset)


def compare_all(args_dict):
    datasets = args_dict.pop('dataset')
    for dataset in datasets:
        compare(**args_dict, dataset=dataset)


if __name__ == '__main__':
    args_dict = vars(get_args())
    mode = args_dict.pop('mode')

    match mode:
        case 'compute':
            compute_all(args_dict)
        case 'compare':
            compare_all(args_dict)
        case _:
            raise ValueError(f"Unknown mode {mode}.")

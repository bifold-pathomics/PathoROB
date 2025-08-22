import os
import glob
import time
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, Normalizer

from pathorob.features.data_manager import FeatureDataManager
from pathorob.features.constants import AVAILABLE_DATASETS
import pathorob.robustness_index.robustness_graphs as robustness_graphs
from pathorob.robustness_index.robustness_index_paired import calc_rob_index_pairs
from pathorob.robustness_index.robustness_index_utils import aggregate_stats, save_total_stats, \
    get_field_names_given_dataset, evaluate_knn_accuracy, get_k_values, \
    save_balanced_accuracies, evaluate_embeddings, calculate_per_class_prediction_stats, \
    calculate_robustness_index_at_k_opt, get_model_colors, get_folder_paths, plot_results_per_model


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

    #required parameters
    parser.add_argument('--model', type=str, default="all", help='Model name or "all" to process all models')
    parser.add_argument('--dataset', type=str, help='Dataset name', choices=AVAILABLE_DATASETS)

    #optional parameters
    parser.add_argument('--k_opt_param', type=int, default=-1, help='Currently, k_opt_param should be set to the default value of -1, which ensures results are produced for all values of k. For future use, this parameter can be set to a specific k value to only report the robustness index for the specified value of k. '                                                                    'Fixed k_opt parameter; if -1, the optimal k value will be optimized based on biological class prediction.')
    parser.add_argument('--max_patches_per_combi', type=int, default=-1, help='Maximum patches per combination. -1 for no limit, or a specific number to limit the dataset size.')
    parser.add_argument('--data_subfolder', type=str, default="default", help='Subfolder specifying a variant of the dataset. The features should be stored in this folder: [embedding_folder]/[data_subfolder].')
    parser.add_argument('--results_folder_root', type=str, default="results/robustness_index", help='Root folder for results.')
    parser.add_argument('--fig_folder_root', type=str, default="results/robustness_index/fig", help='Root folder for figures.')
    parser.add_argument('--embedding_folder', type=str, default="data/features", help='Folder for embeddings. The features should be stored in this folder: [embedding_folder]/[data_subfolder].')
    parser.add_argument('--meta_folder', type=str, default="data/metadata", help='Folder for metadata.')
    parser.add_argument('--compute_bootstrapped_robustness_index', action='store_true', help='Compute bootstrapped robustness index.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for parallel processing.')
    parser.add_argument('--debug_mode', action='store_true', help='Run in debug mode with limited data for faster testing.')
    parser.add_argument('--plot_graphs', type=str2bool, default=True, help='Whether to plot graphs.')

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

    k_values = get_k_values(dataset, opt_k, max_samples_per_group)

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
                                                                                         y_train, y_test, k,
                                                                                         num_workers, knn_distances,
                                                                                         knn_indices)
        acc_score_bio = float(acc_score_bio)
        accuracies_k_bio.append(acc_score_bio)
        aucs_per_class[k] = auc_per_class
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
    save_total_stats(total_stats, meta, dataset, model, results_folder, k_opt, bal_acc_at_k_opt)
    calculate_per_class_prediction_stats(biological_class_field, confounding_class_field, bio_classes, model, meta, aucs_per_class_list, k_opt, results_folder)
    if plot_graphs:
        plot_results_per_model(meta, total_stats, accuracies_bio, k_values, model, results_folder, fig_folder, dataset,
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

    k_opt, bio_class_prediction_results, robustness_metrics_dict = select_optimal_k_value(dataset, model, patch_names, embeddings, meta, results_folder, fig_folder,
                                                                 num_workers = num_workers, compute_bootstrapped_robustness_index=compute_bootstrapped_robustness_index, opt_k=k_opt_param, plot_graphs=plot_graphs)
    print(f"found k_opt {k_opt}")
    return bio_class_prediction_results, robustness_metrics_dict


def calc_rob_index(data_manager, models, dataset, meta, results_folder, fig_folder, num_workers=8, k_opt_param=-1, compute_bootstrapped_robustness_index=False, DBG=False, plot_graphs=True):
    results = {}
    robustness_metrics_dict = {}
    for m,model in enumerate(models):
        print(f"processing model {m+1}/{len(models)}: {model}")
        fn = os.path.join(results_folder, f'frequencies-same-class-{model}.pkl')
        if os.path.exists(fn):
            print(f"model {model}: results already exist --> skipping. Found {fn}")
            continue
        bio_class_prediction_results, robustness_metrics_dict[model] = evaluate_model(dataset, data_manager, model, meta, results_folder, fig_folder, num_workers=num_workers, k_opt_param=k_opt_param,  compute_bootstrapped_robustness_index=compute_bootstrapped_robustness_index, DBG=DBG, plot_graphs=plot_graphs)
        results[model] = bio_class_prediction_results
        robustness_metrics_dict[model] = robustness_metrics_dict
    return results, robustness_metrics_dict


def get_meta(data_manager, dataset, paired_evaluation=True):
    if dataset == "camelyon":
        if paired_evaluation:
            metadata_name = "camelyon_reduced"
        else:
            metadata_name = "camelyon"
    elif dataset == "tcga":
        if paired_evaluation:
            metadata_name = "tcga_2x2"
        else:
            metadata_name = "tcga_4x4"
    elif dataset == "tolkach_esca":
        if paired_evaluation:
            metadata_name = "tolkach_esca_reduced"
        else:
            metadata_name = "tolkach_esca"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    meta = data_manager.load_metadata(metadata_name)
    meta.insert(0, "patch_name", data_manager.compute_ids(meta))
    return meta


def get_bal_acc_values(model, options):
    results_folder = options["results_folder"]
    fn = os.path.join(results_folder, f'bal-acc-bio-{model}.csv')  # get bal_acc for biological classification
    if not os.path.isfile(fn):
        raise ValueError(f'missing bal_acc file {fn}')
    bal_accs_bio = pd.read_csv(fn)
    bal_acc_values = bal_accs_bio.bal_acc.values
    mis = np.isnan(bal_acc_values)
    bal_accs_bio = bal_accs_bio[~mis].reset_index(drop=True)
    bal_acc_values = bal_accs_bio.bal_acc.values
    k_values = bal_accs_bio.k.values
    index_opt = np.argmax(bal_acc_values)
    k_opt = k_values[index_opt]
    return bal_acc_values, k_opt


def report_optimal_k(results_folder, fig_folder, models, options):
    print("Optimal k values")

    plt.figure(figsize=(5, 4))
    opt_k_values = []
    mcolors = get_model_colors(models)
    model_k_opt = {}

    max_bal_acc_values = []
    max_bal_acc_value = {}
    for m, model in enumerate(models):
        fn = os.path.join(results_folder, f'bal-acc-bio-{model}.csv') #get bal_acc for biological classification
        if not os.path.isfile(fn):
            raise ValueError(f'missing accuracy file {fn}')
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
        bal_acc_values, k_opt = get_bal_acc_values(model, options)
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
            fn = os.path.join(results_folder, f'bal-acc-bio-{model}.csv')  # get bal_acc for biological classification
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
    plt.tight_layout()
    plt.savefig(os.path.join(fig_folder, f'optimal-k-values-{model_str}-no-legend.png'), dpi=600)
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
    plt.gcf().set_size_inches(12, 6)
    plt.tight_layout()
    fn = os.path.join(fig_folder, f'optimal-k-values-{model_str}.png')
    plt.savefig(fn, dpi=600)
    print(f"saved optimal k values to {fn}")
    df_optimal_k = pd.DataFrame({'model': models, 'k_opt': opt_k_values, 'max_bal_acc': max_bal_acc_values})
    fn = os.path.join(results_folder, f'optimal-k-values-{model_str}.csv')
    df_optimal_k.to_csv(fn, index=False)
    print(f"saved optimal k values to {fn}")
    return model_k_opt, model_bal_acc_values, max_bal_acc_value


def reduce_dataset(results_folder, dataset, meta, max_patches_per_combi):
    np.random.seed(123)  # always use same fixed seed on purpose here for reproducibility
    meta = meta.sample(frac=1).reset_index(drop=True)
    meta["bio_conf_combi"] = meta["biological_class"] + "-" + meta["medical_center"]
    if max_patches_per_combi > 0:
        nr_org = len(meta)
        grouping_columns = ["subset", "bio_conf_combi"]
        meta = meta.groupby(grouping_columns).head(max_patches_per_combi).reset_index(drop=True)
        print(f"reduced dataset to {max_patches_per_combi} patches per combination from {nr_org} to {len(meta)}")
    fn = os.path.join(results_folder, f'meta-reduced-{dataset}.csv')
    meta.to_csv(fn, index=False)
    print(f"saved reduced meta to {fn}")
    return meta


def results_summary(meta, max_patches_per_combi, results_folder, model_k_opt, median_k_opt, model_bal_acc_values, model_robustness_index, results, dt):
    nr_patches = len(meta)
    result = {}

    print(f"results_summary: models in results: {list(results.keys())}")
    models = results.keys()
    if not model_robustness_index:
        k_opt_used, rob_index_at_k_opt = calculate_robustness_index_at_k_opt(models, results_folder, model_k_opt)
        median_k_opt_used, rob_index_at_median_k_opt = calculate_robustness_index_at_k_opt(models, results_folder, median_k_opt)

    for model in models:
        if results[model] is None:
            k_opt = int(model_k_opt[model])
            if not model_robustness_index is None and model in model_robustness_index:
                k_opt_index = k_opt - 1
                if k_opt_index > len(model_robustness_index[model]) - 1:
                    print(
                        f"found k_opt {k_opt} earlier, but reducing to highest available value: {len(model_robustness_index[model]) - 1} ")
                    k_opt_index = len(model_robustness_index[model]) - 1  # use last available k
                robustness_index = model_robustness_index[model][k_opt_index]
            else:
                robustness_index = -1
            result_string = (
                f"final result max-patches-per-combi {max_patches_per_combi} "
                f"model {model} nr-patches {nr_patches} k_opt {model_k_opt[model]} {k_opt} "
                f"bal_acc {-1} {-1} std {-1} "
                f"robustness index {robustness_index:.3f}"
                f" runtime {dt:.2f} sec, {dt / 60.0:.2f} min")
        else:
            print(results[model])
            model_bal_acc = results[model]["bal_acc_at_k_opt"]
            if model_bal_acc_values:
                bal_acc = np.max(model_bal_acc_values[model].bal_acc.values)
            else:
                bal_acc = -1
            model_k_opt = results[model]["k_opt"]
            index_k_opt_full = model_k_opt - 1 #index in full range 1..k_max
            if model_robustness_index:
                model_rob_index = model_robustness_index[model]
                model_rob_index_at_k_opt = model_rob_index[index_k_opt_full]
                model_rob_index_mean = model_robustness_index[model+"-mean"]
                model_rob_index_std = model_robustness_index[model+"-std"]
                index_median_k_opt = min(len(model_rob_index)-1, median_k_opt - 1)
                print(f"model_k_opt {model_k_opt}  median_k_opt {median_k_opt}")
                bootstrapping_avail = len(model_rob_index_mean) > 1
                rob_index_k_opt, rob_index_k_opt_median = model_rob_index_at_k_opt, model_rob_index[index_median_k_opt]
            else:
                bootstrapping_avail = False
                rob_index_k_opt, rob_index_k_opt_median = rob_index_at_k_opt[model], rob_index_at_median_k_opt[model]


            if bootstrapping_avail:
                mean_std_str = f"robustness_index mean {model_rob_index_mean[index_k_opt_full]:.3f}  std {model_rob_index_std[index_k_opt_full]:.3f}"
                mean_std_str_median = f"robustness_index mean {model_rob_index_mean[index_median_k_opt]:.3f} std {model_rob_index_std[index_median_k_opt]:.3f}"
            else:
                mean_std_str = f"robustness_index mean -1 std -1"
                mean_std_str_median = "robustness_index mean -1 std -1"

            result_string = (f"final result max-patches-per-combi {max_patches_per_combi} "
                             f"model {model} nr-patches {nr_patches} "
                             f"model k_opt {model_k_opt} robustness_index {rob_index_k_opt :.3f} {mean_std_str} "
                             f"median k_opt {median_k_opt} robustness_index {rob_index_k_opt_median:.3f} {mean_std_str_median} "
                             f"bal_acc {bal_acc:.3f} {model_bal_acc:.3f} "                                 
                             f" runtime {dt:.2f} sec, {dt/60.0:.2f} min")

            model_result = {"model": model, "median_k_opt": median_k_opt, "robustness_index_at_median_k_opt": rob_index_k_opt_median, "balanced_accuracy": bal_acc}
            result[model] = model_result

        print(result_string)
        fn = os.path.join(results_folder, f'results-{model}.txt')
        with open(fn,'w') as file:
            file.write(result_string + "\n")
        print(f"wrote result string to {fn}")
        return result


def plot_all_results(models, results_folder, fig_folder, model_k_opt, median_k_opt, dataset, options):
    robustness_metrics, robustness_index = None, None
    plot_per_model_results = True
    if plot_per_model_results:
        fig_folder_per_model = os.path.join(fig_folder, "per-model")
        os.makedirs(fig_folder_per_model, exist_ok=True)
        for model in models:
            robustness_graphs.plot_results(model, results_folder, fig_folder_per_model, model_k_opt)

    plot_all_model_results = True
    if plot_all_model_results:
        robustness_graphs.plot11_performance_robustness_tradeoff(models, options, results_folder, fig_folder, model_k_opt, median_k_opt, dataset=dataset)

        robustness_graphs.plot_4_freq_bio_vs_conf_all_models(models, results_folder, fig_folder)
        robustness_graphs.plot_5_freq_bio_vs_conf_all_models(models, results_folder, fig_folder)
        _                     , _                = robustness_graphs.plot_6_robustness_index_all_models(models, results_folder, fig_folder, model_k_opt, median_k_opt, use_median_k_opt=True, dataset=dataset)
        robustness_metrics, robustness_index = robustness_graphs.plot_6_robustness_index_all_models(models, results_folder, fig_folder, model_k_opt, median_k_opt, use_median_k_opt=False, dataset=dataset) #return this as default below

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
        meta_folder: str = "data/metadata",  # TODO rename
        embedding_folder: str = "data/features",  # TODO rename
        results_folder_root: str = "results/robustness_index",  # TODO rename
        fig_folder_root: str = "results/robustness_index/fig",  # TODO rename
        data_subfolder: str = "default",  # TODO remove; not relevant here
        k_opt_param: int = -1,
        max_patches_per_combi: int = -1,
        compute_bootstrapped_robustness_index: bool = False,
        num_workers: int = 8,
        plot_graphs: bool = True,
        debug_mode: bool = False,
):
    t_start = time.time()

    if model != "all":
        print(f"processing model {model}")
        models = [model]

    # TODO is this the correct logic?
    paired_evaluation = dataset == "tcga"  # default: use paired setup for TCGA, as it has many biological and confounding classes and is not balanced

    options = {"model": model, "max_patches_per_combi": max_patches_per_combi,
              "k_opt_param": k_opt_param, "data_subfolder": data_subfolder, "dataset": dataset, "results_folder_root": results_folder_root,
               "fig_folder_root": fig_folder_root, "paired_evaluation": paired_evaluation, "meta_folder": meta_folder}

    print("using these settings:")
    for param in options:
        print(f"param {param}: {options[param]}")

    DBG=debug_mode
    options["DBG"] = DBG
    results_folder, fig_folder = get_folder_paths(options, dataset)

    if model == "all":
        models = [f.split("/")[-1].replace("frequencies-same-class-", "").replace(".pkl", "") for f in
                  glob.glob(os.path.join(results_folder, "*.pkl"))]

    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(fig_folder, exist_ok=True)

    data_manager = FeatureDataManager(features_dir=embedding_folder, metadata_dir=meta_folder)
    meta = get_meta(data_manager, dataset, options["paired_evaluation"])
    meta = reduce_dataset(results_folder, dataset, meta, max_patches_per_combi=max_patches_per_combi)

    if options["paired_evaluation"]: #calculate robustness index for pairs of 2 bio classes and 2 confounding classes
        robustness_metrics_dict, results = calc_rob_index_pairs(data_manager, models, dataset, meta, results_folder, fig_folder, num_workers=num_workers, k_opt_param=k_opt_param, DBG=DBG, compute_bootstrapped_robustness_index=compute_bootstrapped_robustness_index, plot_graphs=plot_graphs)
    else: #calculate robustness index for any number of bio classes and any number of confounding classes
        robustness_metrics_dict, results = calc_rob_index(data_manager, models, dataset, meta, results_folder, fig_folder, num_workers=num_workers, k_opt_param=k_opt_param, compute_bootstrapped_robustness_index=compute_bootstrapped_robustness_index, DBG=DBG, plot_graphs=plot_graphs)

    if k_opt_param == -1:
        if plot_graphs:
            model_k_opt, model_bal_acc_values, max_bal_acc_value = report_optimal_k(results_folder, fig_folder, models, options)
        recompute_median_k_opt = False
        if recompute_median_k_opt:
            median_k_opt = int(np.median(list(model_k_opt.values())))
        else:
            median_k_opt = get_median_k_opt_given_dataset(dataset)
        print(f"dataset {dataset} found model k_opt {model_k_opt}  median k_opt: {median_k_opt:.2f}")
    else: #fixed k_opt_param
        print(f"using fixed k_opt_param {k_opt_param} for all models")
        model_k_opt = {model: k_opt_param} #use specified value for all plots
        median_k_opt = k_opt_param
        model_bal_acc_values=None

    if plot_graphs:
        robustness_metrics, robustness_index = plot_all_results(models, results_folder, fig_folder, model_k_opt, median_k_opt,
                                                                dataset, options)
    else:
        robustness_metrics = None

    if k_opt_param == -1:
        res = pd.DataFrame({'model': models, 'k_opt': [int(model_k_opt[m]) for m in models], 'bio_prediction_bal_acc': [max_bal_acc_value[m] for m in models]})
        if not robustness_index is None:
            res['robustness_index'] = [robustness_index[m] for m in models]
        if len(models) == 1:
            fn = os.path.join(results_folder, f'results-{dataset}-{models[0]}.csv')
        else:
            fn = os.path.join(results_folder, f'results-{dataset}-{len(models)}-models.csv')

        res.to_csv(fn, index=False)
        print(f"saved results to {fn}")

        if not robustness_metrics is None:
            robustness_graphs.pareto_plot(dataset, models, model_bal_acc_values, robustness_metrics, fig_folder)

    t_end_calc = time.time()
    dt = t_end_calc - t_start
    print(f"calculation time {dt:.2f} seconds = {dt/60:.2f} minutes = {dt/3600:.2f} hours")

    if results:
        result = results_summary(meta, max_patches_per_combi, results_folder, model_k_opt, median_k_opt, model_bal_acc_values, robustness_metrics, results, dt)
        #dict with metric dict per model
        print("final result", result)

    return robustness_metrics_dict


if __name__ == '__main__':
    compute(**vars(get_args()))

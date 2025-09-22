import os
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from pathorob.robustness_index.robustness_index_utils import (
    aggregate_per_combi, calculate_robustness_index_at_k_opt, get_robustness_index_k_range,
    get_optimal_prediction_results_avg_all_datasets, get_optimal_prediction_results_per_dataset,
    get_robustness_results_median_k_opt_per_dataset, get_robustness_results_all_datasets,
    get_robustness_results_per_dataset, get_model_colors, OutputFiles, get_file_path
)


def plot_results(model, results_folder, fig_folder, k_opt_bio):
    fn = results_folder / OutputFiles.FREQUENCIES
    if not os.path.exists(fn):
        print(f"file not found: {fn}")
        return
    results = pickle.load(open(fn, 'rb'))

    stats = results['stats']

    all_bio_classes = results['all_bio_classes']
    all_conf_classes = results['all_conf_classes']

    plt.figure(figsize = (5, 4))
    line_nr = 0
    max_nr_lines_legend=8
    colors = get_colors()

    bc_ri = []
    for bi, bc in enumerate(np.unique(all_bio_classes)):
        if bc in stats:
            fraction_SS, fraction_SO, fraction_OS, fraction_OO = aggregate_per_combi(stats, bc)
            robustness_index_bc = [fraction_SO[k] / (fraction_SO[k] + fraction_OS[k]) for k in range(len(fraction_SO))]
            bc_ri.append(robustness_index_bc[0])
    sortindex = np.argsort(bc_ri)[::-1]

    for bi, bc in enumerate(np.unique(all_bio_classes)[sortindex]):
        if bc in stats:
            line_nr += 1
            fraction_SS, fraction_SO, fraction_OS, fraction_OO = aggregate_per_combi(stats, bc)
            robustness_index_bc = [fraction_SO[k] / (fraction_SO[k] + fraction_OS[k]) for k in range(len(fraction_SO))]
            if line_nr < max_nr_lines_legend or line_nr >= len(all_bio_classes) - max_nr_lines_legend:
                colornr = bi
                if line_nr > max_nr_lines_legend:
                    colornr = len(all_bio_classes)-line_nr + max_nr_lines_legend
                plt.plot(range(len(robustness_index_bc)), robustness_index_bc, label=bc, color = colors[colornr])
            elif line_nr == max_nr_lines_legend:
                    plt.plot(range(len(robustness_index_bc)), robustness_index_bc, label="...", color = colors[bi])
            else:
                plt.plot(range(len(robustness_index_bc)), robustness_index_bc)

    plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')

    plt.title(f"Robustness index per biological class for {model}")
    plt.xlabel("k")
    plt.ylabel("Robustness index")
    plt.tight_layout(pad=2)  # Adds padding around the plot
    fn=os.path.join(fig_folder,f'1-robustness-index-per-bio-class.png')
    plt.savefig(fn, dpi=600)
    print(f"saved robustness index per bio class to {fn}")
    plt.close()

    ri_cc = []
    for ci, cf in enumerate(np.unique(all_conf_classes)):
        if cf in stats:
            line_nr += 1
            fraction_SS, fraction_SO, fraction_OS, fraction_OO = aggregate_per_combi(stats, cf)
            robustness_index_bc = [fraction_SO[k] / (fraction_SO[k] + fraction_OS[k]) for k in range(len(fraction_SO))]
            ri_cc.append(robustness_index_bc[0])
    sortindex = np.argsort(ri_cc)[::-1]

    plt.figure(figsize = (5, 4))

    line_nr = 0
    max_nr_lines_legend=8
    for ci, cf in enumerate(np.unique(all_conf_classes)[sortindex]):
        if cf in stats:
            line_nr += 1
            fraction_SS, fraction_SO, fraction_OS, fraction_OO = aggregate_per_combi(stats, cf)
            robustness_index_bc = [fraction_SO[k] / (fraction_SO[k] + fraction_OS[k]) for k in range(len(fraction_SO))]

            if line_nr < max_nr_lines_legend or line_nr >= len(all_conf_classes) - max_nr_lines_legend:
                colornr = ci
                if line_nr > max_nr_lines_legend:
                    colornr = len(all_conf_classes)-line_nr + max_nr_lines_legend
                plt.plot(range(len(robustness_index_bc)), robustness_index_bc, label=cf, color = colors[colornr])
            elif line_nr == max_nr_lines_legend:
                plt.plot(range(len(robustness_index_bc)), robustness_index_bc, label="...", color = colors[ci])
            else:
                plt.plot(range(len(robustness_index_bc)), robustness_index_bc)
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
    plt.tight_layout(pad=2)  # Adds padding around the plot

    plt.title(f"Robustness index per confounding class for {model}")
    plt.xlabel("k")
    plt.ylabel("Robustness index")
    fn = os.path.join(fig_folder,f'2-robustness-index-per-conf-class.png')
    plt.savefig(fn, dpi=600)
    print(f"saved robustness index per conf class to {fn}")
    plt.close()

    plt.figure()

    robustness_index_bc = stats['robustness_index']

    plt.plot(range(len(robustness_index_bc)), robustness_index_bc)
    plt.legend()
    index_k_opt_bio = int(k_opt_bio - 1)  # k_opt_bio is 1-based, but index is 0-based
    plt.title(f"Robustness index for {model}\n k_opt_bio={k_opt_bio} : robustness index {robustness_index_bc[index_k_opt_bio]:.2f} ")
    plt.plot(k_opt_bio, robustness_index_bc[index_k_opt_bio], 'o',color='#1f77b4')
    plt.gcf().set_size_inches(10, 6)
    plt.xlabel("k")
    plt.ylabel("Robustness index")
    plt.savefig(os.path.join(fig_folder,f'3-robustness-index.png'), dpi=600)
    print(f"saved robustness index to {os.path.join(fig_folder,f'3-robustness-index.png')}")
    plt.close()


    plt.figure(figsize = (10, 6))
    freq_same_bio_class = stats["fraction_SO-norm"]
    freq_same_conf_class = stats["fraction_OS-norm"]

    total = freq_same_bio_class + freq_same_conf_class  # renormalize just SO and OS
    freq_same_bio_class = freq_same_bio_class / total
    freq_same_conf_class = freq_same_conf_class / total

    stepsize=100
    plt.plot(np.arange(1, len(freq_same_bio_class)+1, stepsize), freq_same_bio_class[::stepsize],'g',label="same biological class")
    plt.plot(np.arange(1, len(freq_same_conf_class)+1, stepsize), freq_same_conf_class[::stepsize],'b',label="same confounding class")
    plt.title(f"Frequency of same class for neighbor k\n{model}")
    plt.xlabel("k")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_folder,f'4-freq-same-bio-conf-class-neighbor-k.png'), dpi=600)
    print(f"saved frequency of same class to {os.path.join(fig_folder,f'4-freq-same-bio-conf-class-neighbor-k.png')}")
    plt.close()

    plt.figure(figsize = (10, 6))
    freq_same_bio_class_cum = stats["fraction_SO-cum-norm"]
    freq_same_conf_class_cum = stats["fraction_OS-cum-norm"]
    total = freq_same_bio_class_cum + freq_same_conf_class_cum  # renormalize just SO and OS
    freq_same_bio_class_cum = freq_same_bio_class_cum / total
    freq_same_conf_class_cum = freq_same_conf_class_cum / total


    plt.plot(np.arange(1, len(freq_same_bio_class_cum)+1), freq_same_bio_class_cum,'g',label="same biological class")
    plt.plot(np.arange(1, len(freq_same_conf_class_cum)+1), freq_same_conf_class_cum,'b',label="same confounding class")
    plt.legend()
    plt.title(f"Frequency of same class for k nearest neighbors\n{model}")
    plt.xlabel("k")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_folder,f'5-freq-same-bio-conf-class-knn.png'), dpi=600)
    print(f"saved frequency of same class to {os.path.join(fig_folder,f'5-freq-same-bio-conf-class-knn.png')}")
    plt.close()


def get_colors(min_nr_colors=19):
    return [
        "steelblue",  # Steel Blue
        "darkorange",  # Dark Orange
        "forestgreen",  # Forest Green
        "firebrick",  # Brick Red
        "mediumpurple",  # Medium Purple
        "saddlebrown",  # Chestnut Brown
        "hotpink",  # Rose Pink
        "gray",  # Gray
        "olive",  # Olive
        "cyan",  # Cyan
        "lightcoral",  # Light Coral
        "palegreen",  # Pale Green
        "tomato",  # Vivid Orange
        "lightblue",  # Light Blue
        "red", #"lavender",  # Lavender
        "rosybrown",  # Rosy Brown
        "lightpink",  # Light Pink
        "khaki",  # Khaki
        "black", #"lavenderblush"  # Periwinkle (closest match)
        "goldenrod",
    ]


def plot_4_freq_bio_vs_conf_all_models(models, results_folder, fig_folder, options_subfolder):
    print("plotting freq bio vs conf index for all models")
    plt.figure(figsize=(5, 4))
    mcolors = get_model_colors(models)
    first_bio_value = []
    stepsize = 30
    model_stats = {}
    for m, model in enumerate(models):
        fn = get_file_path(results_folder, model, options_subfolder, OutputFiles.FREQUENCIES)
        results = pickle.load(open(fn, 'rb'))
        stats = results['stats']
        k_range = stats['k']
        fso = stats["fraction_SO-norm"]
        fos = stats["fraction_OS-norm"]
        total = fso + fos #renormalize just SO and OS
        fso = fso / total
        fos = fos / total
        model_stats[model] = {"fso": fso, "fos": fos}
        freq_same_bio_class_neighbor_k = model_stats[model]["fso"]

        if not np.max(k_range) == len(k_range):
            raise ValueError("assuming k_range is a consecutive range starting from 1") #determine the index of k_opt based on k_range below if this happens
        first_bio_value_model = freq_same_bio_class_neighbor_k[0]
        first_bio_value.append(first_bio_value_model)

    first_bio_value = np.array(first_bio_value)
    sorted_indices = np.argsort(first_bio_value)
    sorted_indices = sorted_indices[::-1]
    for i, sorted_index in enumerate(sorted_indices):
        print(f"{i+1}/{len(models)}: {models[sorted_index]} {first_bio_value[sorted_index]:.5f}")

    #first just plot to get label in legend
    for m, model in enumerate([models[k] for k in sorted_indices]):
        freq_same_bio_class = model_stats[model]["fso"]
        freq_same_conf_class = model_stats[model]["fos"]

        plt.plot(range(len(freq_same_bio_class))[::stepsize], freq_same_bio_class[::stepsize], label=f"{model} fraction same biological class", color=mcolors[m])
        if m == 0:
            plt.plot(range(len(freq_same_bio_class))[::stepsize], freq_same_conf_class[::stepsize], '--', label=f"{model} fraction same confounding class", color=mcolors[m])
        else:
            plt.plot(range(len(freq_same_bio_class))[::stepsize], freq_same_conf_class[::stepsize], '--', color=mcolors[m])

    #then plot again in ascending order to ensure visibility of topmost lines
    for m, model in enumerate([models[k] for k in sorted_indices[::-1]]):
        freq_same_bio_class = model_stats[model]["fso"]
        freq_same_conf_class = model_stats[model]["fos"]

        plt.plot(np.arange(1, 1+len(freq_same_bio_class), stepsize), freq_same_bio_class[::stepsize], color=mcolors[m])
        plt.plot(np.arange(1, 1+len(freq_same_conf_class), stepsize), freq_same_conf_class[::stepsize], '--', color=mcolors[m])

    plt.title(f"Frequency of biological vs confounding class\nfor neighbor k")
    plt.xlabel("k")
    plt.ylabel("Frequency")
    plt.xlim([0, len(model_stats[models[0]]["fso"])])
    plt.tight_layout()
    if len(models) > 1:
        fn = os.path.join(fig_folder,f'4-freq-bio-conf-knn-all-models-neighbor-k-no-legend.png')
        plt.savefig(fn, dpi=600)
        print(f"saved freq bio conf index to {fn}")
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
        plt.gcf().set_size_inches(12, 6)
        plt.tight_layout()
        fn = os.path.join(fig_folder,f'4-freq-bio-conf-knn-all-models-neighbor-k.png')
        plt.savefig(fn, dpi=600)
        print(f"saved freq bio conf index to {fn}")
    plt.close()


def plot_5_freq_bio_vs_conf_all_models(models, results_folder, fig_folder, options_subfolder):
    print("plotting freq bio vs conf index for all models")
    plt.figure(figsize=(5, 4))
    mcolors = get_model_colors(models)
    first_bio_value = []
    robustness_index = {}

    model_stats = {}
    for m, model in enumerate(models):
        fn = get_file_path(results_folder, model, options_subfolder, OutputFiles.FREQUENCIES)
        results = pickle.load(open(fn, 'rb'))
        stats = results['stats']
        k_range = stats['k']

        freq_same_bio_class_upto_k = stats["fraction_SO-norm"] / (stats["fraction_SO-norm"] + stats["fraction_OS-norm"] )

        if not np.max(k_range) == len(k_range):
            raise ValueError("assuming k_range is a consecutive range starting from 1") #determine the index of k_opt based on k_range below if this happens
        first_bio_value_model = freq_same_bio_class_upto_k[0]
        first_bio_value.append(first_bio_value_model)

    first_bio_value = np.array(first_bio_value)
    sorted_indices = np.argsort(first_bio_value)
    sorted_indices = sorted_indices[::-1]
    for i, sorted_index in enumerate(sorted_indices):
        print(f"{i+1}/{len(models)}: {models[sorted_index]} {first_bio_value[sorted_index]:.5f}")

    #first just plot to get label in legend
    df_dict = {}
    for m, model in enumerate([models[k] for k in sorted_indices]):
        fn = get_file_path(results_folder, model, options_subfolder, OutputFiles.FREQUENCIES)
        results = pickle.load(open(fn, 'rb'))
        stats = results['stats']

        freq_same_bio_class = stats['fraction_SO-cum-norm']
        freq_same_conf_class = stats['fraction_OS-cum-norm']
        total = freq_same_bio_class + freq_same_conf_class  # renormalize just SO and OS
        freq_same_bio_class = freq_same_bio_class / total
        freq_same_conf_class = freq_same_conf_class / total

        plt.plot(np.arange(1, len(freq_same_bio_class)+1), freq_same_bio_class, label=f"{model} fraction same biological class", color=mcolors[m])
        df_dict["biol "+model] = freq_same_bio_class
        df_dict["conf "+model] = freq_same_conf_class
        if m == 0:
            plt.plot(np.arange(1, len(freq_same_bio_class)+1), freq_same_conf_class, '--', label=f"{model} fraction same confounding class", color=mcolors[m])
            df_dict["k"] = np.arange(1, len(freq_same_bio_class)+1)
        else:
            plt.plot(np.arange(1, len(freq_same_bio_class)+1), freq_same_conf_class, '--', color=mcolors[m])

    #then plot again in ascending order to ensure visibility of topmost lines
    for m, model in enumerate([models[k] for k in sorted_indices[::-1]]):
        fn = get_file_path(results_folder, model, options_subfolder, OutputFiles.FREQUENCIES)
        results = pickle.load(open(fn, 'rb'))
        stats = results['stats']
        k_range = stats["k"]

        freq_same_bio_class = stats['fraction_SO-cum-norm']
        freq_same_conf_class = stats['fraction_OS-cum-norm']
        total = freq_same_bio_class + freq_same_conf_class  # renormalize just SO and OS
        freq_same_bio_class = freq_same_bio_class / total
        freq_same_conf_class = freq_same_conf_class / total

        plt.plot(np.arange(1, len(freq_same_bio_class)+1), freq_same_bio_class, color=mcolors[m])
        plt.plot(np.arange(1, len(freq_same_conf_class)+1), freq_same_conf_class, '--', color=mcolors[m])

    plt.title(f"Frequency of biological vs confounding class\nfor k nearest neighbors ")
    plt.xlabel("k")
    plt.ylabel("Frequency") #the cumsum() above aggregates over neighbors 1-k
    plt.xlim([0,len(freq_same_bio_class)])
    plt.tight_layout()
    if len(models) > 1:
        fn = os.path.join(fig_folder,f'5-freq-bio-conf-knn-all-models-knn-no-legend.png')
        plt.savefig(fn, dpi=600)
        print(f"saved freq bio conf index to {fn}")
    plt.gcf().set_size_inches(12, 6)
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
    plt.tight_layout()
    if len(models) > 1:
        fn = os.path.join(fig_folder,f'5-freq-bio-conf-knn-all-models-knn.png')
        plt.savefig(fn, dpi=600)
        print(f"saved freq bio conf index to {fn}")
    plt.close()

    min_length = np.min([len(df_dict[k]) for k in df_dict.keys()])
    for k in df_dict.keys():
        if len(df_dict[k]) > min_length:
            df_dict[k] = df_dict[k][:min_length]
            print(f"truncated {k} to length {min_length}")

    df = pd.DataFrame(df_dict)
    fn = os.path.join(fig_folder, f'5-frequencies-bio-conf-classes-all-models.csv')
    df.to_csv(fn, index=False)
    print(f"saved frequencies of biological and confounding classes to {fn}")


def plot_robustness_index_values(results_folder, k_opts, rob_index_at_k_opt, models, mcolors, plot_index, add_label, plot_dot, numbers_in_labels, options_subfolder):
    model_robustness_index_k_range = {}
    for m, model in enumerate([models[k] for k in plot_index]):

        k_range, robustness_index, robustness_index_mean, robustness_index_std = get_robustness_index_k_range(model, results_folder, options_subfolder)
        model_robustness_index_k_range[model] = robustness_index
        model_robustness_index_k_range[model+"-mean"] = robustness_index_mean
        model_robustness_index_k_range[model+"-std"] = robustness_index_std

        k_opt = k_opts[model]
        robustness_index_k_opt = rob_index_at_k_opt[model]
        if numbers_in_labels:
            score_str = f"k={k_opt} {robustness_index_k_opt:.3f}"
        else:
            score_str = ""
        if add_label:
            plt.plot(range(len(robustness_index)), robustness_index, label=f"{model} {score_str}",
                     color=mcolors[m])
        else:
            plt.plot(range(len(robustness_index)), robustness_index, color=mcolors[m])

        if plot_dot:
            plt.plot(k_opt, robustness_index_k_opt, 'o', color=mcolors[m])

    return model_robustness_index_k_range


def plot_robustness_with_errorbars(results_folder, models, mcolors, nr_points, fig_folder,
                               use_median_k_opt, sorted_indices, dataset, options_subfolder):
    # now add error bars to this existing plot using std dev
    plt.figure(figsize=(5, 4))
    for m, model in enumerate([models[k] for k in sorted_indices]):
        k_range, robustness_index, _, robustness_index_std = get_robustness_index_k_range(model, results_folder, options_subfolder)
        if robustness_index is None:
            continue
        if len(robustness_index_std) == len(robustness_index):
            plt.errorbar(range(len(robustness_index)), robustness_index, yerr=robustness_index_std,
                         label=f"{model}", color=mcolors[m], capsize=3)
    plt.title(f"Robustness index with error bars\n{dataset}")
    plt.xlabel("k")
    plt.ylabel("Robustness index")
    plt.xlim([0, nr_points])
    plt.tight_layout()
    if use_median_k_opt:
        fn = os.path.join(fig_folder, f'6-robustness-index-all-models-error-bars-median-k_opt-no-legend.png')
    else:
        fn = os.path.join(fig_folder, f'6-robustness-index-all-models-error-bars-model-k_opt-no-legend.png')
    plt.savefig(fn, dpi=600)
    plt.legend()
    plt.gcf().set_size_inches(12, 6)
    plt.tight_layout()

    if use_median_k_opt:
        fn = os.path.join(fig_folder, f'6-robustness-index-all-models-error-bars-median-k_opt.png')
    else:
        fn = os.path.join(fig_folder, f'6-robustness-index-all-models-error-bars-model-k_opt.png')
    plt.savefig(fn, dpi=600)
    print("saved robustness index with error bars to", fn)


def plot_6_robustness_index_all_models(models, results_folder, fig_folder, model_k_opt, median_k_opt, use_median_k_opt, dataset, boostrapped_robustness_index, options_subfolder):
    numbers_in_labels = True
    #plot robustness index for all modest in 1 graph
    print("plotting robustness index for all models")
    plt.figure(figsize=(5, 4))
    mcolors = get_model_colors(models)

    if use_median_k_opt:
        k_opt_used, rob_index_at_k_opt = calculate_robustness_index_at_k_opt(models, results_folder, median_k_opt, options_subfolder)
    else:
        k_opt_used, rob_index_at_k_opt = calculate_robustness_index_at_k_opt(models, results_folder, model_k_opt, options_subfolder)
    rob_index_at_k_opt_array = np.array(list(rob_index_at_k_opt.values()))
    sorted_indices = np.argsort(rob_index_at_k_opt_array)
    sorted_indices = sorted_indices[::-1]

    #first plot to get legend
    plot_index = sorted_indices
    add_label = True
    plot_dot = False
    plot_robustness_index_values(results_folder, k_opt_used, rob_index_at_k_opt, models, mcolors, plot_index, add_label, plot_dot, numbers_in_labels, options_subfolder)

    #then plot top lines last so that these are visible
    plot_index = sorted_indices[::-1]
    add_label=False
    plot_dot = True
    robustness_metrics = plot_robustness_index_values(results_folder, k_opt_used, rob_index_at_k_opt, models, mcolors, plot_index, add_label, plot_dot, numbers_in_labels, options_subfolder)
    nr_points = len(robustness_metrics[models[0]])

    plt.title(f"Robustness index\n{dataset}")
    plt.xlabel("k")
    plt.ylabel("Robustness index") #the cumsum() above aggregates over neighbors 1-k
    plt.xlim([0, nr_points])
    plt.tight_layout()
    k_str = "median-k_opt" if use_median_k_opt else "model-k_opt"
    if len(models) > 1:
        fn = fig_folder / f'6-robustness-index-all-models-no-legend-{k_str}.png'
        plt.savefig(fn, dpi=600)
        # plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
        # plt.gcf().set_size_inches(12, 6)
        # plt.tight_layout()
        # fn = os.path.join(fig_folder,f'6-robustness-index-all-models-{k_str}.png')
        # plt.savefig(fn, dpi=600)
        print(f"saved robustness index to {fn}")

        if boostrapped_robustness_index:
            plot_robustness_with_errorbars(results_folder, models, mcolors, nr_points, fig_folder, use_median_k_opt, sorted_indices, dataset, options_subfolder)
        min_length = np.min([len(robustness_metrics[model]) for model in models])
        remove = []
        for k in robustness_metrics.keys():
            if len(robustness_metrics[k]) > min_length:
                robustness_metrics[k] = robustness_metrics[k][:min_length]
                print(f"truncated {k} to length {min_length}")
            elif len(robustness_metrics[k]) < min_length:
                remove.append(k)
        robustness_metrics = {k: robustness_metrics[k] for k in robustness_metrics.keys() if k not in remove}

        df = pd.DataFrame(robustness_metrics)
        fn = os.path.join(fig_folder, f'6-robustness-index-all-models-{k_str}.csv')
        df.to_csv(fn, index=False)
        print(f"saved robustness index all models to {fn}")
    else:
        fn = fig_folder / f'6-robustness-index-{models[0]}-no-legend-{k_str}.png'
        plt.savefig(fn, dpi=600)
        # plt.gcf().set_size_inches(12, 6)
        # plt.tight_layout()
        # fn = os.path.join(fig_folder,f'6-robustness-index-{models[0]}-{k_str}.png')
        # plt.savefig(fn, dpi=600)
        print(f"saved robustness index single model to {fn}")



    plt.close()
    return robustness_metrics, rob_index_at_k_opt


def plot_10_pareto_plot_avg_all_datasets(datasets, models, options):
    opt_prediction_results = get_optimal_prediction_results_avg_all_datasets(datasets, models, options)
    model_robustness_index, k_range, max_k, avg_median_k_opt_all_datasets, model_robustness_index_at_k_opt = get_robustness_results_all_datasets(datasets, models, options)
    plt.figure(figsize=(11, 6))
    mcolors = get_model_colors(models)

    model_score = {}
    model_opt_index = {}
    all_models = []
    bal_accs = []
    robustness_indices = []
    for m, model in enumerate(models):
        bal_acc_values = opt_prediction_results[model]["bal_acc"]
        k_values = opt_prediction_results[model]["k"]
        k_indices = [k-1 for k in k_values]  #the first robustness_index represents k = 1
        robustness_index = model_robustness_index[model]
        max_k = min(np.max(k_range), len(robustness_index))  #ensure that we do not exceed the length of the robustness index
        k_indices = [k for k in k_indices if k < max_k]  #ensure that we do not exceed the length of the robustness index
        bal_acc_values = bal_acc_values[:len(k_indices)]  #ensure that we do not exceed the length of the robustness index
        avg_rob_bal_acc = (bal_acc_values+robustness_index[k_indices])/2
        opt_index = int(np.argmin(np.abs(k_values - avg_median_k_opt_all_datasets)))
        model_opt_index[model] = opt_index
        max_score = avg_rob_bal_acc[opt_index]
        model_score[model] = max_score
    sort_index = np.argsort([model_score[model] for model in models])
    sort_index = sort_index[::-1]
    opt_k=[]
    for m, model in enumerate([models[k] for k in sort_index]):
        bal_acc_values = opt_prediction_results[model]["bal_acc"]
        k_opt = avg_median_k_opt_all_datasets
        index_k_range = int(np.argmin(np.abs(k_range - avg_median_k_opt_all_datasets)))
        robustness_index = model_robustness_index[model]
        robustness_index_value = robustness_index[index_k_range]
        index_opt = model_opt_index[model]
        bal_acc_value = bal_acc_values[index_opt]
        all_models.append(model)
        bal_accs.append(bal_acc_value)
        opt_k.append(k_opt)
        robustness_indices.append(robustness_index_value)
        plt.plot(robustness_index_value, bal_acc_value, 'o', color=mcolors[m], label=f"{model} k={k_opt} bal_acc {bal_acc_value:.3f} rob {robustness_index_value:.3f}")
    plt.xlabel("Robustness index")
    plt.ylabel("Balanced accuracy")
    plt.title("Optimal tradeoff of prediction performance vs robustness\naveraged over TCGA-Uniform, Camelyon16 and Tolkach-ESCA.")
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
    plt.tight_layout()
    fn = os.path.join(options["figures_dir"], f'10-pareto-plot-avg-3-datasets.png')
    plt.savefig(fn, dpi=600)
    print(f"saved pareto plot to {fn}")
    df = pd.DataFrame({"model": all_models, "k_opt": opt_k, "bal_acc": bal_accs, "robustness_index": robustness_indices})
    fn = os.path.join(options["results_dir"], f'prediction-accuracy-and-robustness-index-averaged-3-datasets.csv')
    df.to_csv(fn, index=False)
    print(f"saved prediction accuracy and robustness index to {fn}")


def plot_10a_pareto_plot_all_datasets(datasets, models, options):
    opt_prediction_results = get_optimal_prediction_results_avg_all_datasets(datasets, models, options)
    model_robustness_index, k_range, max_k, avg_median_k_opt_all_datasets, model_robustness_index_at_k_opt = get_robustness_results_all_datasets(datasets, models, options)
    opt_prediction_results_ds = get_optimal_prediction_results_per_dataset(datasets, models, options)
    model_robustness_index_ds = get_robustness_results_per_dataset(datasets, models, options)
    plt.figure(figsize=(11, 6))
    mcolors = get_model_colors(models)

    model_score = {}
    model_opt_index = {}
    for m, model in enumerate(models):
        bal_acc_values = opt_prediction_results[model]["bal_acc"]
        k_values = opt_prediction_results[model]["k"]
        k_indices = [k-1 for k in k_values]  #the first robustness_index represents k = 1
        robustness_index = model_robustness_index[model]
        max_k = min(np.max(k_range), len(robustness_index))  #ensure that we do not exceed the length of the robustness index
        k_indices = [k for k in k_indices if k < max_k]  #ensure that we do not exceed the length of the robustness index
        bal_acc_values = bal_acc_values[:len(k_indices)]  #ensure that we do not exceed the length of the robustness index
        avg_rob_bal_acc = (bal_acc_values+robustness_index[k_indices])/2
        opt_index = int(np.argmin(np.abs(k_values - avg_median_k_opt_all_datasets)))
        model_opt_index[model] = opt_index
        max_score = avg_rob_bal_acc[opt_index]
        model_score[model] = max_score
    sort_index = np.argsort([model_score[model] for model in models])
    sort_index = sort_index[::-1]
    for m, model in enumerate([models[k] for k in sort_index]):
        bal_acc_values = opt_prediction_results[model]["bal_acc"]
        max_bal_acc = opt_prediction_results[model]["max_bal_acc"]

        robustness_index = model_robustness_index[model]
        index_opt = model_opt_index[model]
        index_k_opt = avg_median_k_opt_all_datasets - 1
        plt.plot(robustness_index[index_k_opt], bal_acc_values[index_opt], 'o', color=mcolors[m], label=f"{model} k={avg_median_k_opt_all_datasets} bal_acc {max_bal_acc:.3f} rob {robustness_index[index_k_opt]:.3f}")

        if m == 0:
            symbols = {"tcga-uniform-subset": "+", "camelyon16": "s", "tolkach-esca": "^"}
            for model in opt_prediction_results_ds:
                for dataset in opt_prediction_results_ds[model]:
                    bal_acc = opt_prediction_results_ds[model][dataset]["max_bal_acc"]
                    k_opt = opt_prediction_results_ds[model][dataset]["k_opt"]
                    robustness_index = model_robustness_index_ds[model][dataset]["robustness_index"]
                    k_range = model_robustness_index_ds[model][dataset]["k_range"]
                    index_k_range = np.where(k_range == k_opt)[0][0]
                    symbol = symbols[dataset]
                    if model == models[0]:
                        plt.plot(robustness_index[index_k_range], bal_acc, symbol, color=mcolors[m],
                                 label=f"{model} {dataset} k={k_opt} bal_acc {bal_acc:.3f} rob {robustness_index[index_k_range]:.3f}")
                    else:
                        plt.plot(robustness_index[index_k_range], bal_acc, symbol, color=mcolors[m])

    plt.xlabel("Robustness index")
    plt.ylabel("Balanced accuracy")
    plt.title(f"Optimal tradeoff of prediction performance vs robustness\naveraged over TCGA-Uniform, Camelyon16 and Tolkach-ESCA")
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
    plt.tight_layout()
    fn = os.path.join(options["figures_dir"], f'10a-pareto-plot-all-3-datasets.png')
    plt.savefig(fn, dpi=600)
    print(f"saved pareto plot to {fn}")


def plot_9_optimal_k_all_datasets(datasets, models, options):
    print("Optimal k values")
    opt_prediction_results = get_optimal_prediction_results_avg_all_datasets(datasets, models, options)

    plt.figure(figsize=(10, 6))
    mcolors = get_model_colors(models)

    sortindex = np.argsort([opt_prediction_results[model]["max_bal_acc"] for model in models])
    sortindex = sortindex[::-1]
    for m, model in enumerate([models[k] for k in sortindex]):
        bal_acc_values = opt_prediction_results[model]["bal_acc"]
        k_values = opt_prediction_results[model]["k"]
        index_opt = opt_prediction_results[model]["index_max_bal_acc"]
        max_bal_acc = opt_prediction_results[model]["max_bal_acc"]
        k_opt = opt_prediction_results[model]["k_opt"]

        plt.plot(k_values, bal_acc_values, color=mcolors[m])
        plt.plot(k_opt, bal_acc_values[index_opt], 'o', color=mcolors[m], label=f"{model} k={k_opt} {max_bal_acc:.3f}")

    plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
    plt.xlabel("k")
    plt.ylabel("Balanced accuracy")
    plt.title(f"Prediction performance against k\naveraged over TCGA-Uniform, Camelyon16 and Tolkach-ESCA")
    plt.tight_layout()
    fn = os.path.join(options["figures_dir"], f'9-optimal-k-avg-3-datasets.png')
    plt.savefig(fn, dpi=600)
    print(f"saved optimal k values to {fn}")


def plot_8_robustness_index_all_datasets(datasets, models, options):
    numbers_in_labels = True
    #plot robustness index for all modest in 1 graph
    print("plotting robustness index for all models")
    plt.figure(figsize=(10, 6))
    mcolors = get_model_colors(models)
    df_lines = {}
    df_dots = {}

    model_robustness_index, k_range, max_k, avg_median_k_opt_all_datasets, model_robustness_index_at_k_opt = get_robustness_results_median_k_opt_per_dataset(
        datasets, models, options)

    models = list(model_robustness_index.keys())
    sort_index = np.argsort([model_robustness_index_at_k_opt[model] for model in models])
    sort_index = sort_index[::-1]
    for m, model in enumerate([models[k] for k in sort_index]):
        k_opt = avg_median_k_opt_all_datasets[model]
        robustness_index_k_opt = model_robustness_index_at_k_opt[model]  # same as robustness_index_mean[k_opt]
        if numbers_in_labels:
            label = f"{model} k={k_opt} {robustness_index_k_opt:.3f}"
        else:
            label = f"{model}"
        nr_points = min(len(k_range), len(model_robustness_index[model]))
        plt.plot(k_range[:nr_points], model_robustness_index[model][:nr_points], label=label, color=mcolors[m])

        df_lines[model] = model_robustness_index[model]

    plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
    plt.title(f"Robustness index for all models\nAveraged over TCGA-Uniform, Camelyon16, Tolkach-ESCA")
    plt.xlabel("k")
    plt.ylabel("Robustness index for k nearest neighbors") #the cumsum() above aggregates over neighbors 1-k
    plt.xlim([0, 50])
    plt.ylim([0.3, 1])
    plt.tight_layout()
    if len(models) > 1:
        median_str = "median-k_opt"
        fn = os.path.join(options["figures_dir"], f'8-robustness-index-all-models-{median_str}-no-dot.png')
        plt.savefig(fn, dpi=600)
        print(f"saved robustness index to {fn}")

        for m, model in enumerate([models[k] for k in sort_index]):
            k_opt = avg_median_k_opt_all_datasets
            robustness_index_k_opt = model_robustness_index_at_k_opt[model] #same as robustness_index_mean[k_opt]
            plt.plot(k_opt[model], robustness_index_k_opt, 'o', color=mcolors[m])
            print(f"plotting dot for model {model} k_opt {k_opt} robustness index {robustness_index_k_opt:.3f}")
            df_dots[model + "-k_opt"] = k_opt
            df_dots[model + "-robustness_index"] = robustness_index_k_opt
        plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
        plt.tight_layout()
        fn = os.path.join(options["figures_dir"], f'8-robustness-index-all-models-{median_str}-with-dot.png')
        plt.savefig(fn, dpi=600)
        print(f"saved robustness index to {fn}")

        min_k_max = np.min([len(v) for v in df_lines.values()])
        df_lines = {k: df_lines[k][:min_k_max] for k in df_lines.keys()}
        df_lines = pd.DataFrame(df_lines)

        fn = os.path.join(options["results_dir"], f'8-avg-robustness-index-all-datasets-{median_str}.csv')
        df_lines.to_csv(fn, index=False)
        print(f"saved robustness index all models to {fn}")
        df_dots = pd.DataFrame(df_dots, index=[0])
        fn = os.path.join(options["results_dir"], f'8-avg-robustness-index-all-datasets-{median_str}-opt-k.csv')
        df_dots.to_csv(fn, index=False)
        print(f"saved robustness index all models dots to {fn}")


def plot11_performance_robustness_tradeoff(models, options, results_folder, fig_folder, model_k_opt, median_k_opt, dataset, options_subfolder):
    for index, model in enumerate(models):
        plt.figure(figsize=(5, 3))
        # Create a second y-axis that shares the same x-axis
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        fn = get_file_path(results_folder, model, options_subfolder, OutputFiles.BALANCED_ACCURACIES)
        bal_accs_bio = pd.read_csv(fn)
        mis = np.isnan(bal_accs_bio.bal_acc.values)
        bal_accs_bio = bal_accs_bio[~mis]
        bal_acc_values = bal_accs_bio.bal_acc.values
        k_values = bal_accs_bio.k.values #not consecutive

        index_opt_k_pred = np.argmax(bal_acc_values)
        k_opt = k_values[index_opt_k_pred]
        model_k_opt[model] = k_opt
        print(f"model {model} optimal k: {k_opt}")

        ax1.plot(k_values, bal_acc_values, 'b-', label=f"Balanced accuracy")
        ax1.plot(k_opt, bal_acc_values[index_opt_k_pred], 'bo')
        ax1.set_xlabel('k')
        ax1.set_ylabel('Balanced accuracy', color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        #k_range: consecutive
        k_range, robustness_index, robustness_index_mean, robustness_index_std = get_robustness_index_k_range(model, results_folder, options_subfolder)
        index_opt_k_rob = np.argmax(robustness_index)

        ax2.plot(k_range, robustness_index, 'g-', label=f"Robustness index")
        ax2.plot(k_range[index_opt_k_rob], robustness_index[index_opt_k_rob], 'go')
        ax2.set_ylabel('Robustness index', color='g')
        ax2.tick_params(axis='y', labelcolor='g')

        plt.title(f"Robustness index and Balanced accuracy vs k\n{dataset}  {model}")
        plt.xlabel("k")
        plt.tight_layout()

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, bbox_to_anchor=(1.08, 0.5), loc='center left')

        plt.gcf().set_size_inches(12, 6)
        plt.tight_layout()
        fn = fig_folder /f"11a-performance-robustness-tradeoff-{dataset}-{model}.png"
        plt.savefig(fn, dpi = 600)
        print(f"saved robustness index to {fn}")
        plt.close()

        plt.figure(figsize=(10, 6))
        # Create a second y-axis that shares the same x-axis

        max_k_value = np.max(k_values)
        k_range = [k for k in k_range if k <= max_k_value]  # ensure that we do not exceed the length of the robustness index
        robustness_index_k_range = [robustness_index[k-1] for k in k_range] #consecutive
        max_k_value = np.max(k_range)
        k_values = [k for k in k_values if k <= max_k_value]  # ensure that we do not exceed the length of the robustness index
        bal_acc_values = bal_acc_values[:len(k_values)]
        robustness_index_k_values = [robustness_index_k_range[k-1] for k in k_values]
        plt.plot(robustness_index_k_values, bal_acc_values, '-')

        acc0 = bal_acc_values[0]
        rob0 = robustness_index_k_range[0]
        plt.plot(rob0, acc0, 'o', color='blue', label=f"k=1 robustness {rob0:.2f} accuracy {acc0:.3f}")


        #now add dots and text boxes for the two optimal k values
        index_opt_k_pred = np.argmax(bal_acc_values)
        k_opt = k_values[index_opt_k_pred]
        robustness_index_k_opt = robustness_index[k_opt-1]
        plt.plot(robustness_index_k_opt, bal_acc_values[index_opt_k_pred], '+', color='blue', label=f"k={k_opt} robustness {robustness_index_k_opt:.2f} accuracy {bal_acc_values[index_opt_k_pred]:.3f}")

        index_opt_k_rob = np.argmax(robustness_index_k_values)
        k_opt_rob = k_values[index_opt_k_rob]
        robustness_index_k_opt_rob = robustness_index_k_values[index_opt_k_rob]
        plt.plot(robustness_index_k_opt_rob, bal_acc_values[index_opt_k_rob], '^', color='blue', label=f"k={k_opt_rob} robustness {robustness_index_k_opt_rob:.2f} accuracy {bal_acc_values[index_opt_k_rob]:.3f}")
        #place text box with the k value and accuracy at the optimal point
        #plt.text(robustness_index_k_opt_rob, bal_acc_values[index_opt_k_rob], f"k={k_opt_rob} {bal_acc_values[index_opt_k_rob]:.3f}", fontsize=10, verticalalignment='bottom', horizontalalignment='right')

        plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')

        plt.xlabel('Robustness index')
        plt.ylabel('Balanced accuracy')
        plt.title(f"Robustness index and Balanced Accuracy\n{dataset}  {model}")
        plt.xlabel("k")
        plt.ylabel("Prediction accuracy")  # the cumsum() above aggregates over neighbors 1-k
        plt.tight_layout()

        plt.gcf().set_size_inches(12, 6)
        plt.tight_layout()
        fn = fig_folder / f"11b-performance-robustness-tradeoff-{dataset}-{model}.png"
        plt.savefig(fn, dpi = 600)
        print(f"saved robustness index to {fn}")
        plt.close()


def pareto_plot(dataset, models, model_bal_acc_values, model_robustness_index, fig_folder):
    plt.figure(figsize=(10, 6))
    mcolors = get_model_colors(models)
    max_bal_acc = [np.max(df.bal_acc.values) for df in model_bal_acc_values.values()]
    sortindex = np.argsort(max_bal_acc)
    sortindex = sortindex[::-1]
    for m, model in enumerate([list(model_bal_acc_values.keys())[k] for k in sortindex]):
        df = model_bal_acc_values[model]
        bal_acc_values = df.bal_acc.values
        k_values = df.k.values
        robustness_index = model_robustness_index[model]
        k_values = [k for k in k_values if k < len(robustness_index)]
        bal_acc_values = bal_acc_values[:len(k_values)]

        rob_values_k = [robustness_index[k-1] for k in k_values]
        plt.plot(rob_values_k, bal_acc_values, color=mcolors[m])
        index_max_bal_acc = np.argmax(bal_acc_values)
        plt.plot(robustness_index[k_values[index_max_bal_acc]], bal_acc_values[index_max_bal_acc], 'o', color=mcolors[m],
                 label = f"{model} k={k_values[index_max_bal_acc]} {bal_acc_values[index_max_bal_acc]:.3f}")

    plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
    plt.xlabel("Robustness index")
    plt.ylabel("Biological prediction accuracy")
    plt.title(f"Robustness index vs accuracy\n{dataset}")
    plt.tight_layout()
    fn = os.path.join(fig_folder, f'7-performance-robustness-pareto-graph.png')
    plt.savefig(fn,  dpi=600)
    print(f"saved robustness index vs accuracy to {fn}")
    plt.close()

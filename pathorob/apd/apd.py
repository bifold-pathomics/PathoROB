import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import trange

from pathorob.features.constants import AVAILABLE_DATASETS
from pathorob.apd.train_model import train_logistic_regression
from pathorob.apd.utils import load_data, get_patches_map_to_split, compute_apd, compute_corrected_scores, load_results


def get_args():
    parser = argparse.ArgumentParser()
    # Required arguments
    parser.add_argument("--model", type=str, required=True)
    # Optional arguments
    parser.add_argument("--datasets", type=str, nargs="+", default=AVAILABLE_DATASETS)
    parser.add_argument("--features_dir", type=str, default="data/features")
    parser.add_argument("--metadata_dir", type=str, default="data/metadata")
    parser.add_argument("--results_dir", type=str, default="results/apd")
    parser.add_argument("--iterations", type=int, default=20)
    return parser.parse_args()


def compute(
        model: str,
        dataset: str,
        features_dir: str = "data/features",
        metadata_dir: str = "data/metadata",
        results_dir: str = "results/apd",
        iterations: int = 20,
):
    output_file = Path(results_dir) / model / dataset

    random.seed(1000)

    print(f"Model: {model}")
    print(f"Dataset: {dataset}")
    print(f"Output files: {str(output_file) + '_summary.json'}, {str(output_file) + '_raw.json'}")
    print(f"Training iterations: {iterations}\n")

    print("Load features and metadata ...")

    # Load features and metadata
    medical_centers, biological_classes, features, data_test_ood, num_splits, num_slides_per_category, num_patches_per_slide, feasible_splits = load_data(model, dataset, features_dir, metadata_dir)

    seeds = random.sample(range(0, 10000), iterations)
    results = {'ID_test_accuracies': [], 'ID_test_accuracy_means': [], 'OOD_test_accuracies': [], 'OOD_test_accuracy_means': []}
    id_test_accuracies, ood_test_accuracies = [[] for _ in range(num_splits)], [[] for _ in range(num_splits)]

    for idx, seed in enumerate(seeds):
        random.seed(seed)

        # For each data resampling repetition, shuffle cases/slides in each features[medical_center][bio_class] list
        if dataset == "tolkach_esca": random_train_test_split = random.randint(0, len(feasible_splits['train'][medical_centers[0]]) - 1), random.randint(0, len(feasible_splits['train'][medical_centers[1]]) - 1)
        for i, center in enumerate(medical_centers):
            if dataset == "tolkach_esca":
                train_cases = feasible_splits['train'][center][random_train_test_split[i]]
                test_cases = feasible_splits['test'][center][random_train_test_split[i]]
            for j in range(len(biological_classes)):
                dataChunkedBySlides = [features[i][j][k:k + num_patches_per_slide] for k in range(0, len(features[i][j]), num_patches_per_slide)]
                random.shuffle(dataChunkedBySlides)
                if dataset == "tolkach_esca":
                    case_order = [dataChunkedBySlides[k][0][3] for k in range(len(dataChunkedBySlides))]
                    for test_case in list(set(test_cases) & set(case_order)):
                        dataChunkedBySlides.append(dataChunkedBySlides.pop([dataChunkedBySlides[k][0][3] for k in range(len(dataChunkedBySlides))].index(test_case)))
                features[i][j] = [item for slide in dataChunkedBySlides for item in slide]

        for split in trange(num_splits, desc=f"Training repetition {idx + 1}/{len(seeds)}", leave=True):
            # Train LR model for each split and store the accuracies
            
            # First, construct training set, validation set, and test sets
            split_map, max_train_slides = get_patches_map_to_split(dataset, split, num_patches_per_slide)

            data_train = [features[i][j][:num_patches] for i, j, num_patches in split_map]
            data_train = [feature_tuple for class_features in data_train for feature_tuple in class_features]

            data_validation = [features[i][j][num_patches:num_patches + int(num_patches / max_train_slides)] for i, j, num_patches in split_map]
            data_validation = [feature_tuple for class_features in data_validation for feature_tuple in class_features]

            idxTest = (max_train_slides + 1) * num_patches_per_slide
            data_test_id = [features[i][j][idxTest:] for i, j in [(a, b) for a in range(len(medical_centers)) for b in range(len(biological_classes))]]
            data_test_id = [feature_tuple for class_features in data_test_id for feature_tuple in class_features]

            data_train = [(patch, bio_class) for patch, _, bio_class, _ in data_train]
            data_validation = [(patch, bio_class) for patch, _, bio_class, _ in data_validation]
            data_test_id = [(patch, bio_class) for patch, _, bio_class, _ in data_test_id]

            train_x, train_y = zip(*data_train)
            val_x, val_y = zip(*data_validation)
            test_x_id, test_y_id = zip(*data_test_id)
            test_x_ood, test_y_ood = zip(*[(patch, bio_class) for patch, _, bio_class, _ in data_test_ood])

            # Then train model and store ID/OOD test set accuracies
            _, _, test_scores = train_logistic_regression(train_x, train_y, val_x, val_y, [test_x_id, test_x_ood], [test_y_id, test_y_ood])
            id_test_accuracies[split].append(test_scores[0])
            ood_test_accuracies[split].append(test_scores[1])

    # Save accuracies in file
    results['ID_test_accuracies'] = id_test_accuracies
    results['ID_test_accuracy_means'] = [np.mean(lst) for lst in id_test_accuracies]
    results['OOD_test_accuracies'] = ood_test_accuracies
    results['OOD_test_accuracy_means'] = [np.mean(lst) for lst in ood_test_accuracies]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_file) + '_raw.json', 'w') as file:
        json.dump({k: v for k, v in results.items() if "means" not in k}, file, indent=4)

    print(f"\nID accuracy means on {dataset} for each split: {results['ID_test_accuracy_means']}")
    print(f"OOD accuracy means on {dataset} for each split: {results['OOD_test_accuracy_means']}")

    # Compute APD for given dataset
    apds = {'apd_id': np.mean(compute_apd(results['ID_test_accuracies'])), 'apd_ood': np.mean(compute_apd(results['OOD_test_accuracies']))}

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_file) + '_summary.json', 'w') as file:
        json.dump(apds | {k: v for k, v in results.items() if "means" in k}, file, indent=4)

    print(f"\nIn-domain APD: {np.round(apds['apd_id'] * 100, 3)}% and out-of-domain APD: {np.round(apds['apd_ood'] * 100, 3)}% for model '{model}' on '{dataset}'.")


if __name__ == '__main__':
    arguments = vars(get_args())
    print(f"Start APD calculation for model: {arguments['model']} on datasets: {arguments['datasets']}\n")
    for dataset in arguments['datasets']:
        args = {**arguments, "dataset": dataset}
        args.pop("datasets")
        compute(**args)
    
    print(f"\nCompute APD over all specified datasets: {arguments['datasets']}")
    res_df = load_results(arguments['results_dir'], arguments['model'], arguments['datasets'])

    # Compute corrected scores
    id_res_df = compute_corrected_scores(res_df[res_df["domain"] == "ID"])
    ood_res_df = compute_corrected_scores(res_df[res_df["domain"] == "OOD"])

    # Get APD with 95% confidence intervals
    stats_ID = id_res_df["corrected_scores"].agg(["mean", "sem"])
    stats_OOD = ood_res_df["corrected_scores"].agg(["mean", "sem"])
    apd_id, ci_id = stats_ID["mean"], stats.t.ppf(0.975, df=len(id_res_df)) * stats_ID["sem"]
    apd_ood, ci_ood = stats_OOD["mean"], stats.t.ppf(0.975, df=len(ood_res_df)) * stats_OOD["sem"]

    apds_all_datasets = {'apd_id': apd_id, 'ci_id': ci_id, 'apd_ood': apd_ood, 'ci_ood': ci_ood}

    (Path(arguments['results_dir']) / arguments['model']).mkdir(parents=True, exist_ok=True)
    with open(Path(arguments['results_dir']) / arguments['model'] / 'aggregated_summary.json', 'w') as file:
        json.dump(apds_all_datasets, file, indent=4)
    
    print(f"In-domain APD: {np.round(apd_id * 100, 3)}% (confidence interval: +-{np.round(ci_id * 100, 3)}) and out-of-domain APD: {np.round(apd_ood * 100, 3)}% (confidence interval: +-{np.round(ci_ood * 100, 3)})")

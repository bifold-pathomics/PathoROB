import os
import sys
import numpy as np
import random
import csv
import json
import argparse
from tqdm import trange

from utils.load_data import load_data
from utils.build_splits import getPatchesMapToSplit
from utils.train_model import train_logistic_regression

parser = argparse.ArgumentParser()
parser.add_argument("--features_dir", type=str, required=True)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--iterations", type=int, required=False, default=20)
args = parser.parse_args()
features_dir = args.features_dir
dataset = args.dataset
iterations = args.iterations
output_file = f"./results/accuracies_{dataset}.json"

random.seed(1000)

print(f"Feature directory: {features_dir}")
print(f"Dataset: {dataset}")
print(f"Output file: {output_file}")
print(f"Training iterations: {iterations}\n")

print("Load features and metadata ...")

# Load features and metadata
medical_centers, biological_classes, features, data_test_ood, num_splits, num_slides_per_category, num_patches_per_slide, tolkach_splits = load_data(dataset, features_dir)

seeds = random.sample(range(0, 10000), iterations)
results = {'ID_test_accuracies': [], 'ID_test_accuracy_means': [], 'OOD_test_accuracies': [], 'OOD_test_accuracy_means': []}
id_test_accuracies, ood_test_accuracies = [[] for _ in range(num_splits)], [[] for _ in range(num_splits)]

for idx, seed in enumerate(seeds):
    random.seed(seed)

    # For each data resampling repetition, shuffle cases/slides in each features[medical_center][bio_class] list
    if dataset == "tolkach_esca": random_train_test_split = random.randint(0, len(tolkach_splits['train'][medical_centers[0]]) - 1), random.randint(0, len(tolkach_splits['train'][medical_centers[1]]) - 1)
    for i, center in enumerate(medical_centers):
        if dataset == "tolkach_esca": 
            train_cases = tolkach_splits['train'][center][random_train_test_split[i]]
            test_cases = tolkach_splits['test'][center][random_train_test_split[i]]
        for j in range(len(biological_classes)):
            dataChunkedBySlides = [features[i][j][k:k + num_patches_per_slide] for k in range(0, len(features[i][j]), num_patches_per_slide)]
            random.shuffle(dataChunkedBySlides)
            if dataset == "tolkach_esca":
                case_order = [dataChunkedBySlides[k][0][3] for k in range(len(dataChunkedBySlides))]
                for test_case in list(set(test_cases) & set(case_order)):
                    dataChunkedBySlides.append(dataChunkedBySlides.pop([dataChunkedBySlides[k][0][3] for k in range(len(dataChunkedBySlides))].index(test_case)))
            features[i][j] = [item for slide in dataChunkedBySlides for item in slide]

    for split in trange(num_splits, desc=f"Training repetition {idx + 1}/{len(seeds)}", leave=True, file=sys.stdout):
        # Train LR model for each split and store the accuracies

        # First, construct training set, validation set, and test sets
        split_map, max_train_slides = getPatchesMapToSplit(dataset, split, num_patches_per_slide)
        
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

os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, 'w') as file:
    json.dump(results, file, indent=4)

print(f"\nID accuracy means on {dataset} for each split: {results['ID_test_accuracy_means']}")
print(f"OOD accuracy means on {dataset} for each split: {results['OOD_test_accuracy_means']}")

# Compute APD for given dataset
# To compute APD over all datasets run this script for each dataset, then run apd_all_datasets.py

def compute_scores(accuracies):
    scores = np.asarray(accuracies)  # Shape: (num_splits, iterations)
    scores = (scores[1:] / scores[0]).mean(axis=0) - 1
    return scores

apd_id = np.mean(compute_scores(results['ID_test_accuracies']))
apd_ood = np.mean(compute_scores(results['OOD_test_accuracies']))

print(f"\nIn-domain APD: {np.round(apd_id * 100, 3)}% and out-of-domain APD: {np.round(apd_ood * 100, 3)}% for {dataset}")
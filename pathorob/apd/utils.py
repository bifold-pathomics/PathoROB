import json
from pathlib import Path
import importlib.resources as pkg_resources

import numpy as np
import pandas as pd
from tqdm import tqdm

from pathorob import resources
from pathorob.features.data_manager import FeatureDataManager


def load_data(model, dataset, features_dir, metadata_dir):
    """
    Load the extracted features and corresponding metadata for downstream experiment.

    Args:
        model (str): The selected foundation model name.
        dataset (str): The selected dataset name; either `camelyon`, `tcga`, or `tolkach_esca`.
        features_dir (str): Source directory of the features.
        features_dir (str): Source directory of the metadata.

    Returns:
        medical_centers_id (list of str): In-domain medical centers for the downstream experiment.
        biological_classes (list of str): Biological classes for the downstream experiment.
        features (nested_list[medical_center_idx][biological_class_idx] of tuples (feature, medical_center_idx, biological_class_idx, slide_id)): Nested list of features with corresponding metadata.
        data_test_ood (list of tuples (feature, medical_center_idx, biological_class_idx, slide_id)): Out-of-domain test features with corresponding metadata.
        num_splits (int): Number of splits for downstream experiment.
        num_slides_per_category (int): Number of slides per category (med_center-bio_class-combination) for downstream experiment.
        num_patches_per_slide (int): Number of patches per slide for downstream experiment.
        feasible_splits (dict['train' | 'test'] or None): Feasible slide-level train/test splits for downstream experiments. Only necessary for tolkach downstream experiment. Since slides in camelyon and tcga can only contribute patches to one biological class, no further computation/handling is necessary. 
    """
    manager = FeatureDataManager(features_dir=features_dir, metadata_dir=metadata_dir)

    medical_centers, biological_classes = [], []
    if dataset == "camelyon":
        metadata_name = "camelyon"
        medical_centers_id = ["RUMC", "UMCU"]
        medical_centers_ood = ["CWZ", "RST", "LPON"]
        biological_classes = ["normal", "tumor"]
        num_splits, num_slides_per_category, num_patches_per_slide = 8, 17, 300
        feasible_splits = None
    elif dataset == "tcga":
        metadata_name = "tcga_4x4"
        medical_centers_id = ["Asterand", "Christiana Healthcare", "Roswell Park", "University of Pittsburgh"]
        medical_centers_ood = ["Cureline", "Greater Poland Cancer Center", "International Genomics Consortium", "Johns Hopkins"]
        biological_classes = ["Breast_invasive_carcinoma", "Colon_adenocarcinoma", "Lung_adenocarcinoma", "Lung_squamous_cell_carcinoma"]
        num_splits, num_slides_per_category, num_patches_per_slide = 7, 12, 30
        feasible_splits = None
    elif dataset == "tolkach_esca":
        metadata_name = "tolkach_esca"
        medical_centers_id = ["VALSET2_WNS", "VALSET4_CHA_FULL"]
        medical_centers_ood = ["VALSET1_UKK", "VALSET3_TCGA"]
        biological_classes = ["TUMOR", "MUSC_PROP", "SH_OES", "SH_MAG", "REGR_TU", "ADVENT"]
        num_splits, num_slides_per_category, num_patches_per_slide = 4, 9, 100
        with pkg_resources.files(resources).joinpath("tolkach_splits.json").open("r") as f:
            feasible_splits = json.load(f)
    else:
        raise ValueError(f"Dataset '{dataset}' not supported.")

    # Load metadata
    metadata = manager.load_metadata(metadata_name)
    metadata_id = metadata[metadata["subset"] == "ID"]
    metadata_ood = metadata[metadata["subset"] == "OOD"]

    # Load ID features and reorganize them for split creation
    features_tmp = manager.load_features(model, dataset, metadata_id)
    features_tmp = list(zip(
        features_tmp,
        metadata_id['medical_center'].map(lambda x: medical_centers_id.index(x)),  # Medical center index
        metadata_id['biological_class'].map(lambda x: biological_classes.index(x)),  # Biological class index
        metadata_id['slide_id']
    ))
    features = [[[] for _ in range(len(biological_classes))] for _ in range(len(medical_centers_id))]
    for patch, medical_center, bio_class, slide_id in features_tmp:
        features[medical_center][bio_class].append((patch, medical_center, bio_class, slide_id))

    # Load OOD features
    data_test_ood = manager.load_features(model, dataset, metadata_ood)
    data_test_ood = list(zip(
        data_test_ood,
        metadata_ood['medical_center'].map(lambda x: medical_centers_ood.index(x)),  # Medical center index
        metadata_ood['biological_class'].map(lambda x: biological_classes.index(x)),  # Biological class index
        metadata_ood['slide_id']
    ))

    return medical_centers_id, biological_classes, features, data_test_ood, num_splits, num_slides_per_category, num_patches_per_slide, feasible_splits


def get_patches_map_to_split(dataset, split, num_patches_per_slide):
    """
    Calculate number of training patches per category (med_center-bio_class-combination) for a given split.

    Args:
        dataset (str): The selected dataset; either `camelyon`, `tcga`, or `tolkach_esca`.
        split (int: [0, ..., splits-1]): The split for which the numbers are calculated.
        num_patches_per_slide (int): Number of patches per slide for downstream experiment.

    Returns:
        list of tuples (i, j, num_paches): List of numbers of training patches (num_patches) per category: med_center(i)-bio_class(j)-combination.
        int: Maximum number of training slides per category.
    """
    if dataset == "camelyon":
        tss0_pairs = [(0, 0, (7 - split) * num_patches_per_slide), (0, 1, (7 + split) * num_patches_per_slide)]
        tss1_pairs = [(1, 0, (7 + split) * num_patches_per_slide), (1, 1, (7 - split) * num_patches_per_slide)]
        return sorted(tss0_pairs + tss1_pairs), 14

    elif dataset == "tcga":
        diag_pairs = [(i, j, (split + 2) * num_patches_per_slide) for i in range(4) for j in range(4) if i == j]
        inv_diag_pairs = [(i, j, (1 if split % 2 == 1 else (2 if split < 3 else 0)) * num_patches_per_slide) for i in
                          range(4) for j in range(4) if i + j == 3]
        rest_pairs = [(i, j, (2 if split < 2 else (1 if split < 5 else 0)) * num_patches_per_slide) for i in range(4)
                      for j in range(4) if i != j and i + j != 3]
        return sorted(diag_pairs + inv_diag_pairs + rest_pairs), 8

    elif dataset == "tolkach_esca":
        tss0_pairs = [(0, j, (3 - split) * num_patches_per_slide) for j in range(3)] + [
            (0, j, (3 + split) * num_patches_per_slide) for j in range(3, 6)]
        tss1_pairs = [(1, j, (3 + split) * num_patches_per_slide) for j in range(3)] + [
            (1, j, (3 - split) * num_patches_per_slide) for j in range(3, 6)]
        return sorted(tss0_pairs + tss1_pairs), 6

    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def compute_apd(accuracies):
    scores = np.asarray(accuracies)  # Shape: (num_splits, iterations)
    scores = (scores[1:] / scores[0]).mean(axis=0) - 1
    return scores


def compute_corrected_scores(res_df):
    # Correct scores following Masson & Loftus
    res_df = res_df.copy()
    res_df.insert(len(res_df.columns), 'corrected_scores', 0.0)
    overall_mean = res_df["scores"].mean()
    for dataset in res_df["dataset"]:
        res_df_dataset = res_df[res_df["dataset"] == dataset]
        dataset_mean = res_df_dataset["scores"].mean()
        corrected_scores = res_df_dataset['scores'] + overall_mean - dataset_mean
        res_df.loc[res_df_dataset.index, "corrected_scores"] = corrected_scores
    return res_df


def load_results(results_dir, model, dataset_names):
    # Load accuracies from files and compute scores
    results_dir = Path(results_dir)
    res_df = pd.DataFrame()
    for dataset in dataset_names:
        with open(results_dir / model / f"{dataset}_raw.json", 'r') as file:
            results = json.load(file)
            scores_ID = compute_apd(results['ID_test_accuracies'])
            scores_OOD = compute_apd(results['OOD_test_accuracies'])
            res_df = pd.concat([
                res_df,
                pd.DataFrame.from_dict({
                    'domain': ["ID"] * len(scores_ID),
                    'dataset': [dataset] * len(scores_ID),
                    'scores': scores_ID
                }),
                pd.DataFrame.from_dict({
                    'domain': ["OOD"] * len(scores_OOD),
                    'dataset': [dataset] * len(scores_OOD),
                    'scores': scores_OOD
                })
            ], ignore_index=True)
    return res_df

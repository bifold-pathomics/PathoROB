from itertools import combinations

import pandas as pd
import numpy as np
from tqdm import tqdm

from pathorob.features.data_manager import FeatureDataManager


def all_subsets(s, min_size=0, max_size=None):
    """
    Generate all subsets of set `s` with cardinality between min_size and max_size (inclusive).

    Args:
        s (set): The input set.
        min_size (int): Minimum cardinality of subsets (inclusive). Default is 0.
        max_size (int): Maximum cardinality of subsets (inclusive). Default is size of set.

    Returns:
        list of sets: List of subsets satisfying the cardinality constraints.
    """
    s = list(s)
    n = len(s)
    if max_size is None:
        max_size = n
    if max_size > n:
        max_size = n
    return [set(c) for r in range(min_size, max_size + 1) for c in combinations(s, r)]


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
        tolkach_splits (dict['train' | 'test']): Feasible slide-level train/test splits for tolkach downstream experiment. Since slides in camelyon and tcga can only contribute to one biological class, no further computation is necessary. 
    """
    manager = FeatureDataManager(features_dir=features_dir, metadata_dir=metadata_dir)

    medical_centers, biological_classes = [], []
    if dataset == "camelyon":
        metadata_name = "camelyon"
        medical_centers_id = ["RUMC", "UMCU"]
        medical_centers_ood = ["CWZ", "RST", "LPON"]
        biological_classes = ["normal", "tumor"]
        num_splits, num_slides_per_category, num_patches_per_slide = 8, 17, 300
        tolkach_splits = {}
    elif dataset == "tcga":
        metadata_name = "tcga_4x4"
        medical_centers_id = ["Asterand", "Christiana Healthcare", "Roswell Park", "University of Pittsburgh"]
        medical_centers_ood = ["Cureline", "Greater Poland Cancer Center", "International Genomics Consortium", "Johns Hopkins"]
        biological_classes = ["Breast_invasive_carcinoma", "Colon_adenocarcinoma", "Lung_adenocarcinoma", "Lung_squamous_cell_carcinoma"]
        num_splits, num_slides_per_category, num_patches_per_slide = 7, 12, 30
        tolkach_splits = {}
    elif dataset == "tolkach_esca":
        metadata_name = "tolkach_esca"
        medical_centers_id = ["VALSET2_WNS", "VALSET4_CHA_FULL"]
        medical_centers_ood = ["VALSET1_UKK", "VALSET3_TCGA"]
        biological_classes = ["TUMOR", "MUSC_PROP", "SH_OES", "SH_MAG", "REGR_TU", "ADVENT"]
        num_splits, num_slides_per_category, num_patches_per_slide = 4, 9, 100

        # Get feasible splits
        sel_metadata = manager.load_metadata(metadata_name)
        tolkach_splits = {'train': {medical_centers_id[0]: [], medical_centers_id[1]: []}, 'test': {medical_centers_id[0]: [], medical_centers_id[1]: []}}

        for sel_center in medical_centers_id:
            sub_metadata = sel_metadata[sel_metadata["medical_center"] == sel_center]
            case_stats = pd.crosstab(sub_metadata["slide_id"], sub_metadata["biological_class"])
            case_stats.insert(len(case_stats.columns), "SUM", case_stats.sum(axis=1))
            case_stats = pd.concat([case_stats, pd.DataFrame.from_dict(
                {key: [val] for key, val in case_stats.sum(axis=0).to_dict().items()},
            ).rename(index={0: "SUM"})], axis=0)
            
            case_idx_set = set(range(len(case_stats) - 1))
            case_idx_subsets = all_subsets(case_idx_set, min_size=2, max_size=12 if sel_center == medical_centers_id[0] else 5)
            
            for idx, test_set_idx in tqdm(list(enumerate(case_idx_subsets))):
                test_set = case_stats.iloc[list(test_set_idx)]
                if (test_set[biological_classes].sum(axis=0) == 200).all():
                    tolkach_splits['test'][sel_center].append(case_stats.index[list(test_set_idx)].tolist())
                    tolkach_splits['train'][sel_center].append(case_stats.index[list(case_idx_set - test_set_idx)].tolist())
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

    return medical_centers_id, biological_classes, features, data_test_ood, num_splits, num_slides_per_category, num_patches_per_slide, tolkach_splits


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

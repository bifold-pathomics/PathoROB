def getPatchesMapToSplit(dataset, split, num_patches_per_slide):
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
        inv_diag_pairs = [(i, j, (1 if split % 2 == 1 else (2 if split < 3 else 0)) * num_patches_per_slide) for i in range(4) for j in range(4) if i + j == 3]
        rest_pairs = [(i, j, (2 if split < 2 else (1 if split < 5 else 0)) * num_patches_per_slide) for i in range(4) for j in range(4) if i != j and i + j != 3]
        return sorted(diag_pairs + inv_diag_pairs + rest_pairs), 8
        
    else:
        tss0_pairs = [(0, j, (3 - split) * num_patches_per_slide) for j in range(3)] + [(0, j, (3 + split) * num_patches_per_slide) for j in range(3, 6)]
        tss1_pairs = [(1, j, (3 + split) * num_patches_per_slide) for j in range(3)] + [(1, j, (3 - split) * num_patches_per_slide) for j in range(3, 6)]
        return sorted(tss0_pairs + tss1_pairs), 6
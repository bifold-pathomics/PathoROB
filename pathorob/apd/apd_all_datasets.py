import argparse
import json
from typing import List
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from pathorob.features.constants import AVAILABLE_DATASETS
from pathorob.apd.utils import compute_apd


def get_args():
    parser = argparse.ArgumentParser()
    # Required arguments
    parser.add_argument("--model", type=str, required=True)
    # Optional arguments
    parser.add_argument("--datasets", nargs="+", default=AVAILABLE_DATASETS)
    parser.add_argument("--results_dir", type=str, default="results/apd")
    return parser.parse_args()


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
        with open(results_dir / model / f"accuracies_{dataset}.json", 'r') as file:
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


def compute(
        model: str,
        datasets: List[str] = AVAILABLE_DATASETS,
        results_dir: str = "results/apd",
):
    # Load accuracies for specified datasets
    res_df = load_results(results_dir, model, datasets)

    # Compute corrected scores
    id_res_df = compute_corrected_scores(res_df[res_df["domain"] == "ID"])
    ood_res_df = compute_corrected_scores(res_df[res_df["domain"] == "OOD"])

    # Get APD with 95% confidence intervals
    stats_ID = id_res_df["corrected_scores"].agg(["mean", "sem"])
    stats_OOD = ood_res_df["corrected_scores"].agg(["mean", "sem"])
    apd_id, ci_id = stats_ID["mean"], stats.t.ppf(0.975, df=len(id_res_df)) * stats_ID["sem"]
    apd_ood, ci_ood = stats_OOD["mean"], stats.t.ppf(0.975, df=len(ood_res_df)) * stats_OOD["sem"]

    print(f"In-domain APD: {np.round(apd_id * 100, 3)}% (confidence interval: +-{np.round(ci_id * 100, 3)}) and out-of-domain APD: {np.round(apd_ood * 100, 3)}% (confidence interval: +-{np.round(ci_ood * 100, 3)}) over datasets: {datasets}")


if __name__ == '__main__':
    compute(**vars(get_args()))

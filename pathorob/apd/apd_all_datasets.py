import numpy as np
import scipy
import json
import pandas as pd
import argparse

from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument("--datasets", nargs="+", required=False, default=['camelyon', 'tcga', 'tolkach_esca'])
args = parser.parse_args()
datasets = args.datasets

def compute_scores(accuracies):
    scores = np.asarray(accuracies)  # Shape: (num_splits, num_repetitions)
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
    
def load_results(dataset_names):
    # Load accuracies from files and compute scores
    res_df = pd.DataFrame()
    for dataset in dataset_names:
        with open('./results/accuracies_' + dataset + '.json', 'r') as file:
            results = json.load(file)
            scores_ID = compute_scores(results['ID_test_accuracies'])
            scores_OOD = compute_scores(results['OOD_test_accuracies'])
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

# Load accuracies for specified datasets
res_df = load_results(datasets)

# Compute corrected scores
id_res_df = compute_corrected_scores(res_df[res_df["domain"] == "ID"])
ood_res_df = compute_corrected_scores(res_df[res_df["domain"] == "OOD"])

# Get APD with 95% confidence intervals
stats_ID = id_res_df["corrected_scores"].agg(["mean", "sem"])
stats_OOD = ood_res_df["corrected_scores"].agg(["mean", "sem"])
apd_id, ci_id = stats_ID["mean"], stats.t.ppf(0.975, df=len(id_res_df)) * stats_ID["sem"]
apd_ood, ci_ood = stats_OOD["mean"], stats.t.ppf(0.975, df=len(ood_res_df)) * stats_OOD["sem"]

print(f"In-domain APD: {np.round(apd_id * 100, 3)}% (confidence interval: +-{np.round(ci_id * 100, 3)}) and out-of-domain APD: {np.round(apd_ood * 100, 3)}% (confidence interval: +-{np.round(ci_ood * 100, 3)}) over datasets: {datasets}")
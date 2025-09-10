# PathoROB

## 1) Installation

```shell
conda create -n "pathorob" python=3.10 -y
conda activate pathorob
pip install -r requirements.txt
```

> [!Note]
> To ensure that the conda environment does not contain any user-specific site packages (e.g., from ~/.local/lib), run `export PYTHONNOUSERSITE=1` after activating your environment.


## 2) Feature Extraction

```
python3 -m pathorob.features.extract_features \
--model uni2h_clsmean \
--model_args '{"hf_token": "<TOKEN>"}'
```

- Results: `data/features/uni2h_clsmean`
- Further arguments: `pathorob/features/extract_features.py`

## 3) Benchmark: Quickstart

### (a) Robustness index

```
python3 -m pathorob.robustness_index.robustness_index \
--model uni2h_clsmean \
--dataset { camelyon OR tcga OR tolkach_esca }
```

- Results: `results/robustness_index`
- Further arguments: `pathorob/robustness_index/robustness_index.py`

### (b) Average performance drop (APD)

```
python3 -m pathorob.apd.apd_per_dataset \
--model uni2h_clsmean \
--dataset { camelyon OR tcga OR tolkach_esca }
```

- Results: `results/apd`
- Further arguments: `pathorob/apd/apd_per_dataset.py`

### (c) Clustering score

```
python3 -m pathorob.clustering_score.clustering_score \
--model uni2h_clsmean \
--dataset { camelyon OR tcga OR tolkach_esca }
```

- Results: `results/clustering_score`
- Further arguments: `pathorob/clustering_score/clustering_score.py`


## 4) Clustering Analysis

### Clustering score computation

To evaluate the clustering performance, navigate to directory `PathoROB/pathorob/clustering_score/` and run the following script:

```python clustering_score.py --model <your FM> --dataset <your dataset>```

where `<your dataset>` must match one of PathoROB’s datasets, i.e., `camelyon`, `tolkach_esca`, or `tcga`. `<your FM>`is the name of the foundation model for which the clustering score will be evaluated. This name must match the name of the folder in `feature_dir` that holds the feature representations of this foundation model. `clustering_score.py` also provides other optional arguments that let you further modify your experiments. Please see `python clustering_score.py --help` for more information.

In its default version, this script first selects the number of clusters by maximizing the silhouette score, then performs clustering 50 times, and finally calculates the average clustering score and its standard deviation. The results are printed to the standard output. Additionally, the results are saved in the following CSV files:
* `<2xbiological_class-2xmedical_center-combination>_SilhouetteScores.csv` (shape: `[2, 29]`):
    Contains silhouette scores for each tested number of clusters.
    - `[0,:]`: tested values of K (in this case, 2 to 30)
    - `[1,:]`: corresponding silhouette scores
* `<2xbiological_class-2xmedical_center-combination>_ARI.csv` (shape: `[50, 3]`):
    Contains the adjusted Rand indices (ARI) for each of the 50 repetitions.
    - `[:,0]`: ARI for biological labels
    - `[:,1]`: ARI for medical center labels
    - `[:,2]`: clustering scores

The evaluation of the clustering score is conducted for each 2x2 pairing of two biological classes and two medical centers separately which is indicated in the results files `<2xbiological_class-2xmedical_center-combination>`.
* expected run time: depends on `<your dataset>`. Per `<2xbiological_class-2xmedical_center-combination>` the expected run time is less than 5 minutes on a MacBook Pro 3112 (M4). The number of `<2xbiological_class-2xmedical_center-combination>` differs for each dataset: `camelyon`: 1, `tolkach_esca`: 45, `tcga`: 36.

Results location:
```
PathoROB/results/clustering_score/<your FM>/<your dataset>/
```

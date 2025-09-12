# PathoROB

[Preprint](https://arxiv.org/abs/2507.17845) | [Hugging Face](https://huggingface.co/collections/bifold-pathomics/pathorob-6899f50a714f446d0c974f87) |  [User Guide](#user-guide) | [Licenses](#licenses) | [Cite](#how-to-cite)

**PathoROB is a benchmark for the robustness of pathology foundation models (FMs) to non-biological medical center differences.**

![PathoROB pipeline](docs/pathorob_pipeline.png)

PathoROB contains **four datasets** covering 28 biological classes from 34 medical centers and **three metrics**:
1. **Robustness Index**: Measures the ability of an FM to capture biological features while ignoring
non-biological features.
2. **Average Performance Drop (APD)**: Measures the impact of non-biological features on the generalization performance of downstream models.
3. **Clustering Score**: Measures the effect of non-biological features on the quality of k-means clusters.

![PathoROB overview](docs/pathorob_overview.png)

## Leaderboard: Robustness Index

| Rank | Foundation Model | TCGA 2x2 | Camelyon | Tolkach ESCA | Average (↓) |
|------|:-----------------|---------:|---------:|-------------:|------------:|
| 1    | Virchow2         |    0.848 |    0.806 |        0.955 |       0.870 |
| 2    | CONCHv1.5        |    0.853 |    0.774 |        0.951 |       0.859 |
| 3    | Atlas            |    0.846 |    0.785 |        0.938 |       0.856 |
| 4    | Virchow          |    0.789 |    0.751 |        0.932 |       0.824 |
| 5    | H0-mini          |    0.816 |    0.718 |        0.932 |       0.822 |
| 6    | H-optimus-0      |    0.838 |    0.705 |        0.918 |       0.820 |
| 7    | CONCH            |    0.847 |    0.662 |        0.951 |       0.820 |
| 8    | UNI2-h           |    0.836 |    0.544 |        0.923 |       0.768 |
| 9    | MUSK             |    0.749 |    0.467 |        0.928 |       0.715 |
| 10   | HIPT             |    0.622 |    0.649 |        0.726 |       0.666 |
| 11   | Prov-GigaPath    |    0.768 |    0.399 |        0.754 |       0.640 |
| 12   | Kaiko ViT-B/8    |    0.788 |    0.147 |        0.896 |       0.610 |
| 13   | UNI              |    0.775 |    0.145 |        0.902 |       0.607 |
| 14   | RetCCL           |    0.609 |    0.318 |        0.878 |       0.602 |
| 15   | CTransPath       |    0.677 |    0.106 |        0.872 |       0.552 |
| 16   | Kang-DINO        |    0.685 |    0.043 |        0.832 |       0.520 |
| 17   | RudolfV          |    0.611 |    0.184 |        0.695 |       0.497 |
| 18   | Phikon           |    0.648 |    0.011 |        0.795 |       0.485 |
| 19   | Phikon-v2        |    0.648 |    0.019 |        0.768 |       0.478 |
| 20   | Ciga             |    0.523 |    0.135 |        0.693 |       0.450 |

All results were computed as part of our benchmarking study. For details as well as for the APD and clustering score results, please check our [preprint](https://arxiv.org/abs/2507.17845). 

> [!Note]
> If you want your model to be added, please [contact](#contact) us.


## User guide

### Installation

```shell
git clone https://github.com/bifold-pathomics/PathoROB.git
cd PathoROB
conda create -n "pathorob" python=3.10 -y
conda activate pathorob
pip install -r requirements.txt
```

> [!Note]
> To ensure that the conda environment does not contain any user-specific site packages (e.g., from `~/.local/lib`), run `export PYTHONNOUSERSITE=1` after activating your environment.


### Feature extraction

```shell
python3 -m pathorob.features.extract_features --model uni2h_clsmean --model_args '{"hf_token": "<TOKEN>"}'
```

For feature extraction, ~100K images (~2GB) will be downloaded from [Hugging Face](https://huggingface.co/collections/bifold-pathomics/pathorob-6899f50a714f446d0c974f87). 

- Results: `data/features/uni2h_clsmean`
- Datasets: Per default, features for all PathoROB datasets will be extracted (`camelyon`, `tcga`, `tolkach_esca`). To select any subset of these, use `--datasets <dataset1> ...`.
- Further arguments: `pathorob/features/extract_features.py`

### Benchmark metrics

#### (1) Robustness Index

```
python3 -m pathorob.robustness_index.robustness_index \
--model uni2h_clsmean \
--dataset { camelyon OR tcga OR tolkach_esca }
```

- Results: `results/robustness_index`
- Further arguments: `pathorob/robustness_index/robustness_index.py`

#### (2) Average Performance Drop (APD)

```shell
python3 -m pathorob.apd.apd --model uni2h_clsmean
```

- Results: `results/apd`
   - `{model}/{dataset}_raw.json` per {dataset}:
      - In-/out-of-domain accuracies per split and trail.
   - `{model}/{dataset}_summary.json` per {dataset}:
      - In-/out-of-domain APDs for the specific {dataset}.
      - In-/out-of-domain accuracy means per split averaged over trails.
   - `{model}/aggregated_summary.json`:
      - In-/out-of-domain APDs with 95% confidence intervals over all specified datasets.
- Further arguments: `pathorob/apd/apd.py`

#### (3) Clustering Score

```shell
python3 -m pathorob.clustering_score.clustering_score --model uni2h_clsmean
```

- Results: `results/clustering_score`
   - `{model}/{dataset}/results_summary.json`:
     - Summary of the results including the final clustering score. 
   - `{model}/{dataset}/aris.csv`:
     - Raw Adjusted Rand Index (ARI) and clustering scores for all for repetitions.
   - `{model}/{dataset}/silhouette_scores.csv`:
     - Raw silhouette scores to select the optimal K for clustering.
- Further arguments: `pathorob/clustering_score/clustering_score.py`

### Adding your own model

1. Create a new Python file in `pathorob.models`.
2. Import the `ModelWrapper` from `pathorob.models.utils`.
3. Define a new model wrapper class and implement the required functions (see template below).
   - Examples: `pathorob.models.uni` or `pathorob.models.phikon`.
4. Add your model wrapper to the `load_model` function in `pathorob.models.__init__` and choose a `model_name`.
5. Run the feature extraction script using your `model_name`.
   - `python3 -m pathorob.features.extract_features --model <model_name>`

```python
from pathorob.models.utils import ModelWrapper


class MyModelWrapper(ModelWrapper):

    def __init__(self, ...):
        """
        Optional: Define custom arguments that can be passed to the model in the
        extract_features entrypoint via `--model_args` as a dictionary. 
        """

    def get_model(self):
        """
        :return: A model object (e.g., `torch.nn.Module`) that has an `eval()` and a
        `to(device)` method.
        """

    def get_preprocess(self):
        """
        Preprocessing to apply to raw PIL images before passing the data to the model.

        :return: A function or an executable object (e.g., `torchvision.transforms.Compose`)
            that accepts a `PIL.Image` as input and returns what the `extract` function
            needs for feature extraction. Note that the result will be batched by the
            default `collate_fn` of a torch DataLoader (`torch.utils.data.DataLoader`).
        """

    def extract(self, data) -> torch.Tensor:
        """
        Feature extraction step for preprocessed and batched image data.

        :param data: (batch_size, ...) A batch of preprocessed image data. The images were
            preprocessed individually via `get_preprocess()` and batched via the default
            `collate_fn` of a torch DataLoader  (`torch.utils.data.DataLoader`).
        :return: (batch_size, feature_dim) A torch Tensor containing the extracted features.
        """
```

## Latest updates

- September 2025: PathoROB is now available on Hugging Face and GitHub. 

## Licenses

The PathoROB datasets were subsampled from public sources. Therefore, we redistribute each PathoROB dataset under the license of its original data source. You can run PathoROB on any subset of datasets with licenses suitable for your application.

- **Camelyon**:
  - Source: [CAMELYON16](https://camelyon16.grand-challenge.org/) and [CAMELYON17](https://camelyon17.grand-challenge.org/Home/)
  - License: CC0 1.0 (Public Domain)
- **TCGA**:
  - Source: [TCGA-UT](https://zenodo.org/records/5889558)
  - License: CC-BY-NC-SA 4.0 (Non-Commercial Use)
- **Tolkach ESCA**
  - Source: https://zenodo.org/records/7548828
  - License: CC-BY-SA 4.0
  - Comment: This license was granted by the author specifically for PathoROB.

## Acknowledgements

We want to thank the authors of the original datasets for making their data publicly available.

## Contact

If you have questions or feedback, please contact:
- Jonah Kömen (koemen@tu-berlin.de)
- Edwin D. de Jong (edwin.dejong@aignostics.com)
- Julius Hense (j.hense@tu-berlin.de)

## How to cite

If you find **PathoROB** useful, please cite our preprint:
```
@article{koemen2025pathorob,
    title={Towards Robust Foundation Models for Digital Pathology},
    author={K{\"o}men, Jonah and de Jong, Edwin D and Hense, Julius and Marienwald, Hannah and Dippel, Jonas and Naumann, Philip and Marcus, Eric and Ruff, Lukas and Alber, Maximilian and Teuwen, Jonas and others},
    journal={arXiv preprint arXiv:2507.17845},
    year={2025}
}
```

Please also cite the source publications of _all_ PathoROB datasets that you use:

- **Camelyon** (Source: [CAMELYON16](https://camelyon16.grand-challenge.org/) and [CAMELYON17](https://camelyon17.grand-challenge.org/Home/), License: CC0 1.0)
```
@article{bejnordi2017camelyon16,
    title={Diagnostic Assessment of Deep Learning Algorithms for Detection of Lymph Node Metastases in Women With Breast Cancer},
    author={Ehteshami Bejnordi, Babak and Veta, Mitko and Johannes van Diest, Paul and van Ginneken, Bram and Karssemeijer, Nico and Litjens, Geert and van der Laak, Jeroen A. W. M. and and the CAMELYON16 Consortium},
    journal={JAMA},
    year={2017},
    volume={318},
    number={22},
    pages={2199-2210},
    doi={10.1001/jama.2017.14585}
}
```
```
@article{bandi19camelyon17,
    title={From Detection of Individual Metastases to Classification of Lymph Node Status at the Patient Level: The CAMELYON17 Challenge},
    author={Bándi, Péter and Geessink, Oscar and Manson, Quirine and Van Dijk, Marcory and Balkenhol, Maschenka and Hermsen, Meyke and Ehteshami Bejnordi, Babak and Lee, Byungjae and Paeng, Kyunghyun and Zhong, Aoxiao and Li, Quanzheng and Zanjani, Farhad Ghazvinian and Zinger, Svitlana and Fukuta, Keisuke and Komura, Daisuke and Ovtcharov, Vlado and Cheng, Shenghua and Zeng, Shaoqun and Thagaard, Jeppe and Dahl, Anders B. and Lin, Huangjing and Chen, Hao and Jacobsson, Ludwig and Hedlund, Martin and Çetin, Melih and Halıcı, Eren and Jackson, Hunter and Chen, Richard and Both, Fabian and Franke, Jörg and Küsters-Vandevelde, Heidi and Vreuls, Willem and Bult, Peter and van Ginneken, Bram and van der Laak, Jeroen and Litjens, Geert},
    journal={IEEE Transactions on Medical Imaging}, 
    year={2019},
    volume={38},
    number={2},
    pages={550-560},
    doi={10.1109/TMI.2018.2867350}
}
```

- **TCGA** (Source: [TCGA-UT](https://zenodo.org/records/5889558), License: CC-BY-NC-SA 4.0)
```
@article{komura22tcga-ut,
    title={Universal encoding of pan-cancer histology by deep texture representations},
    author={Daisuke Komura and Akihiro Kawabe and Keisuke Fukuta and Kyohei Sano and Toshikazu Umezaki and Hirotomo Koda and Ryohei Suzuki and Ken Tominaga and Mieko Ochi and Hiroki Konishi and Fumiya Masakado and Noriyuki Saito and Yasuyoshi Sato and Takumi Onoyama and Shu Nishida and Genta Furuya and Hiroto Katoh and Hiroharu Yamashita and Kazuhiro Kakimi and Yasuyuki Seto and Tetsuo Ushiku and Masashi Fukayama and Shumpei Ishikawa},
    journal={Cell Reports},
    year={2022},
    volume={38},
    number={9},
    pages={110424},
    doi={https://doi.org/10.1016/j.celrep.2022.110424}
}
```

- **Tolkach ESCA** (Source: https://zenodo.org/records/7548828, License: CC-BY-SA 4.0)
```
@article{tolkach2023esca,
    title={Artificial intelligence for tumour tissue detection and histological regression grading in oesophageal adenocarcinomas: a retrospective algorithm development and validation study},
    author={Tolkach, Yuri and Wolgast, Lisa Marie and Damanakis, Alexander and Pryalukhin, Alexey and Schallenberg, Simon and Hulla, Wolfgang and Eich, Marie-Lisa and Schroeder, Wolfgang and Mukhopadhyay, Anirban and Fuchs, Moritz and others},
    journal={The Lancet Digital Health},
    year={2023},
    volume={5},
    number={5},
    pages={e265--e275},
    publisher={Elsevier}
}
```

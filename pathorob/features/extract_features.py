import argparse
import json
from typing import List, Dict, Any

import pandas as pd
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

from pathorob.features.data_manager import FeatureDataManager
from pathorob.features.constants import AVAILABLE_DATASETS
from pathorob.models import load_model


def get_args():
    parser = argparse.ArgumentParser()
    # Required arguments
    parser.add_argument(
        "--model", type=str, required=True,
        help="Name of the model to extract features for. Model names are specified in 'pathorob.models.__init__'."
    )
    # Optional arguments
    parser.add_argument(
        "--model_args", type=json.loads, default={},
        help="Optional arguments to be passed to the model initialization (e.g., weights path of download token)."
    )
    parser.add_argument(
        "--datasets", nargs="+", default=AVAILABLE_DATASETS,
        help=f"PathoROB datasets to extract features for. Available datasets: {AVAILABLE_DATASETS}."
    )
    parser.add_argument(
        "--features_dir", type=str, default="data/features",
        help="Directory to save extracted features."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for the dataloader."
    )
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="Number of workers for the dataloader."
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run model on"
    )
    return parser.parse_args()


def preprocess_fn(data, preprocess):
    data["image"] = list(map(preprocess, data["image"]))
    return data


def extend_metadata(all_metadata, new_metadata):
    for key, val in new_metadata.items():
        if key in all_metadata:
            all_metadata[key].extend(val)
        else:
            all_metadata[key] = val
    return all_metadata


def extract_features(model_wrapper, data_manager, model_name, dataset_name, dataset_path, batch_size, num_workers, device):

    # Prepare data loader
    print(f"Loading 'PathoROB-{dataset_name}' dataset...")
    dataset = load_dataset(dataset_path, split="train")
    dataset.set_transform(lambda x: preprocess_fn(x, model_wrapper.get_preprocess()))
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    # Prepare model
    model_wrapper.get_model().eval()
    model_wrapper.get_model().to(device)

    # Run feature extraction
    features = []
    metadata = {}

    with torch.no_grad():
        for batch in tqdm(
                data_loader, leave=False,
                desc=f"Extracting 'PathoROB-{dataset_name}' features for model '{model_name}'"
        ):
            # Extract features
            batch_images = batch.pop("image").to(device)
            batch_features = model_wrapper.extract(batch_images).detach().cpu()

            if len(batch_features.shape) != 2 or batch_features.shape[0] > batch_size:
                raise ValueError(
                    f"Extracted features have unexpected shape: {batch_features.shape}. "
                    f"Expected: (batch_size, feature_dim)."
                )

            # Collect features and metadata
            features.append(batch_features)
            metadata = extend_metadata(metadata, batch)

            # Delete temporary data (this allows the current process to be killed)
            del batch_images
            del batch

        features = torch.cat(features)
        metadata = pd.DataFrame(metadata)

    data_manager.save_features(model_name, dataset_name, features, metadata)


def compute(
        model: str,
        model_args: Dict[str, Any],
        datasets: List[str] = AVAILABLE_DATASETS,
        features_dir: str = "data/features",
        batch_size: int = 32,
        num_workers: int = 0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    # Load model and context
    print(f"Loading '{model}' model...")
    model_wrapper = load_model(model_name=model, model_args=model_args)
    data_manager = FeatureDataManager(features_dir=features_dir)
    print(f"Feature extraction will be run on device: '{device}'")
    device = torch.device(device)

    for dataset_name in datasets:
        extract_features(
            model_wrapper=model_wrapper,
            data_manager=data_manager,
            model_name=model,
            dataset_name=dataset_name,
            dataset_path=f"bifold-pathomics/PathoROB-{dataset_name}",
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )
        print(f"Saved 'PathoROB-{dataset_name}' features for model '{model}' to: {features_dir}.")


if __name__ == "__main__":
    compute(**vars(get_args()))

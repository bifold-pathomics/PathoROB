import argparse
import json

import pandas as pd
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

from pathorob.models import load_model
from pathorob.features.data_manager import FeatureDataManager


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, required=True,
        help="PathoROB datasets to extract features for."
    )
    parser.add_argument(
        "--model_args", type=json.loads, default={},
        help="Optional arguments to be passed to the model."
    )
    parser.add_argument(
        "--save_dir", type=str, default="./data/features",
        help="Directory to save extracted features."
    )
    parser.add_argument(
        "--subsets", nargs="+", default=["camelyon", "tcga", "tolkach_esca"],
        help="PathoROB datasets to extract features for."
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


def extract_features(model_wrapper, data_manager, subset, batch_size, num_workers, device):

    # Prepare data loader
    print(f"Loading 'PathoROB-{subset}' dataset...")
    dataset = load_dataset(f"bifold-pathomics/PathoROB-{subset}", split="train")
    dataset.set_transform(lambda x: preprocess_fn(x, model_wrapper.get_preprocess()))
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    # Prepare model
    model_wrapper.get_model().eval()
    model_wrapper.get_model().to(device)

    # Run feature extraction
    features = []
    metadata = {}

    with torch.no_grad():
        for batch in tqdm(data_loader, leave=False, desc=f"Extracting 'PathoROB-{subset}' features"):

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

    data_manager.save_features(subset, features, metadata)


def main():
    args = get_args()

    # Load model and context")
    print(f"Loading '{args.model_name}' model...")
    model_wrapper = load_model(model_name=args.model_name, model_args=args.model_args)
    data_manager = FeatureDataManager(args.save_dir)
    device = torch.device(args.device)

    for subset in args.subsets:
        extract_features(
            model_wrapper=model_wrapper,
            data_manager=data_manager,
            subset=subset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        print(f"Saved 'PathoROB-{subset}' features to: {args.save_dir}.")


if __name__ == "__main__":
    main()

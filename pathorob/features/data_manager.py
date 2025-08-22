import warnings
from pathlib import Path

import pandas as pd
import numpy as np


METADATA_COLUMNS = ["subset", "slide_id", "patch_id", "biological_class", "medical_center"]


class FeatureDataManager:

    def __init__(self, features_dir="data/features", metadata_dir="data/metadata", chunk_key="medical_center"):
        self.features_dir = Path(features_dir)
        self.metadata_dir = Path(metadata_dir)
        self.chunk_key = chunk_key

    def compute_ids(self, metadata):
        return metadata[["slide_id", "patch_id"]].astype(str).agg("-".join, axis=1)

    def get_available_models(self):
        return sorted([
            model_dir.stem for model_dir in self.features_dir.iterdir()
            if model_dir.is_dir()
        ])

    def get_available_datasets(self, model_name):
        model_dir = self.features_dir / model_name
        return sorted([
            dataset_dir.stem for dataset_dir in model_dir.iterdir()
            if dataset_dir.is_dir()
        ])

    def get_available_metadata(self):
        return sorted([
            metadata_dir.stem for metadata_dir in self.metadata_dir.iterdir()
            if metadata_dir.is_file()
        ])

    def save_features(self, model_name, dataset_name, features, metadata):
        # Create dataset path
        dataset_dir = self.features_dir / model_name / dataset_name
        dataset_dir.mkdir(exist_ok=True, parents=True)
        # Assign unique identifiers to the features
        metadata = metadata.copy()
        metadata.insert(0, "id", self.compute_ids(metadata))
        # Divide the features into chunks
        for chunk_id in metadata[self.chunk_key].unique():
            # Check for existing features
            chunk_file = dataset_dir / f"{chunk_id.replace(' ', '_')}.npz"
            if chunk_file.is_file():
                features_dict = dict(np.load(chunk_file))
            else:
                features_dict = {}
            # Save chunk
            metadata_chunk = metadata[metadata[self.chunk_key] == chunk_id]
            features_dict.update({id: features[idx] for idx, id in metadata_chunk["id"].items()})
            np.savez(chunk_file, **features_dict)

    def load_features(self, model_name, dataset_name, metadata):
        # Load dataset path
        dataset_dir = self.features_dir / model_name / dataset_name
        if not dataset_dir.is_dir():
            raise ValueError(
                f"Cannot find features for model '{model_name}' dataset '{dataset_name}' at: '{dataset_dir}'."
            )
        # Recover identifiers of the features
        metadata = metadata.copy()
        metadata.insert(0, "id", self.compute_ids(metadata))
        # Iterate over the chunks of features
        features_dict = {}
        for chunk_id in metadata[self.chunk_key].unique():
            metadata_chunk = metadata[metadata[self.chunk_key] == chunk_id]
            # Load chunk of features
            chunk_file = dataset_dir / f"{chunk_id.replace(' ', '_')}.npz"
            if not chunk_file.is_file():
                raise ValueError(
                    f"Cannot find features for chunk '{chunk_id}' at: '{chunk_file}'."
                )
            features_chunk = dict(np.load(chunk_file))
            features_dict.update({id: features_chunk[id] for id in metadata_chunk["id"].unique()})
        # Sort features according to given metadata structure
        features = np.stack([features_dict[id] for id in metadata["id"]])
        return features

    def load_metadata(self, metadata_name):
        metadata_path = self.metadata_dir / f"{metadata_name}.csv"
        if not metadata_path.is_file():
            raise ValueError(f"Cannot find metadata '{metadata_name}' at: '{metadata_path}'.")
        metadata = pd.read_csv(metadata_path, index_col=None, dtype=str)
        missing_cols = [col for col in metadata.columns if col not in METADATA_COLUMNS]
        if len(missing_cols) > 0:
            warnings.warn(f"Missing columns in metadata '{metadata_name}': {missing_cols}")
        return metadata

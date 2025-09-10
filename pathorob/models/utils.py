from abc import ABC, abstractmethod

import torch


IMAGENET_NORM = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class ModelWrapper(ABC):

    @abstractmethod
    def get_model(self):
        """
        :return: A model object (e.g., `torch.nn.Module`) that has an `eval()` and a `to(device)` method.
        """
        pass

    @abstractmethod
    def get_preprocess(self):
        """
        Preprocessing to apply to raw PIL images before passing the data to the model.

        :return: A function or an executable object (e.g., `torchvision.transforms.Compose`) that accepts a `PIL.Image`
            as input and returns what the `extract` function needs for feature extraction. Note that the result will
            be batched by the default `collate_fn` of a torch DataLoader (`torch.utils.data.DataLoader`).
        """
        pass

    @abstractmethod
    def extract(self, data) -> torch.Tensor:
        """
        Feature extraction step for preprocessed and batched image data.

        :param data: (batch_size, ...) A batch of preprocessed image data. The images were preprocessed individually
            via `get_preprocess()` and batched via the default `collate_fn` of a torch DataLoader
            (`torch.utils.data.DataLoader`).
        :return: (batch_size, feature_dim) A torch Tensor containing the extracted features.
        """
        pass

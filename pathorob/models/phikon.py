import torch
import torchvision.transforms as transforms
from transformers import AutoModel

from pathorob.models.utils import ModelWrapper, IMAGENET_NORM


class Phikonv2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("owkin/phikon-v2", force_download=True)

    def forward_features(self, *args, **kwargs):
        outputs = self.encoder(*args, **kwargs)
        return outputs.last_hidden_state


class Phikonv2ModelWrapper(ModelWrapper):

    def __init__(self):
        self.model = Phikonv2()

    def get_model(self):
        return self.model

    def get_preprocess(self):
        return transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(*IMAGENET_NORM),
            ]
        )

    def extract(self, data):
        # Concatenate class token and mean of patch tokens
        features = self.model.forward_features(data)
        cls_token = features[:, 0, :]
        if hasattr(self.model, 'num_reg_tokens') and self.model.num_reg_tokens > 0:
            patch_tokens = features[:, self.model.num_reg_tokens+1:]
        else:
            patch_tokens = features[:, 1:, :]
        features = torch.cat([cls_token, torch.mean(patch_tokens, dim=1)], dim=1)
        return features

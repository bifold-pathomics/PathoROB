from huggingface_hub import login
import torch
import torchvision.transforms as transforms
import timm

from pathorob.models.utils import ModelWrapper, IMAGENET_NORM


class UNI2hModelWrapper(ModelWrapper):

    def __init__(self, hf_token):
        login(token=hf_token, add_to_git_credential=False)
        timm_kwargs = {
            'img_size': 224,
            'patch_size': 14,
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5,
            'embed_dim': 1536,
            'mlp_ratio': 2.66667 * 2,
            'num_classes': 0,
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked,
            'act_layer': torch.nn.SiLU,
            'reg_tokens': 8,
            'dynamic_img_size': True
        }
        self.model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
        self.model.eval()

    @staticmethod
    def get_preprocess():
        return transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(*IMAGENET_NORM),
            ]
        )

    def get_model(self):
        return self.model

    def extract(self, data):
        features = self.model(data)
        return features

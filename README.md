# PathoROB

## 1) Installation

Requirements for feature extraction:
- numpy
- pandas

 
- tqdm
- torch
- torchvision
- datasets
- huggingface_hub
- timm

Additional requirements for PathoROB repository:
- pandas==2.2.3
- numpy==2.1.3


- scipy==1.14.1
- scikit-learn==1.5.2
- matplotlib==3.9.2
- seaborn==0.13.2


## 2) Feature Extraction

How to run feature extraction:

```
python3 -m pathorob.features.extract_features \
--model_name uni2-h \
--model_args '{"hf_token": "<TOKEN>"}'
```

For further user arguments, see `pathorob/features/extract_features.py`.
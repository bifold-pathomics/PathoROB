from pathorob.models.uni import UNI2hModelWrapper
from pathorob.models.phikon import Phikonv2ModelWrapper


def load_model(model_name, model_args):
    if model_name == "uni2h_clsmean":
        model_class = UNI2hModelWrapper
    elif model_name == "phikonv2_clsmean":
        model_class = Phikonv2ModelWrapper
    else:
        raise ValueError(f"Model not implemented: '{model_name}'.")
    model_wrapper = model_class(**model_args)
    return model_wrapper

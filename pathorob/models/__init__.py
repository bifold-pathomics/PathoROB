from pathorob.models.uni import UNI2hModelWrapper


def load_model(model_name, model_args):
    if model_name == "uni2-h":
        model_class = UNI2hModelWrapper
    else:
        raise ValueError(f"Model not implemented: '{model_name}'.")
    model_wrapper = model_class(**model_args)
    return model_wrapper

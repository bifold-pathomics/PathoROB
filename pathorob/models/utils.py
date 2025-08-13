from abc import ABC, abstractmethod


IMAGENET_NORM = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class ModelWrapper(ABC):

    @staticmethod
    @abstractmethod
    def get_preprocess():
        pass

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def extract(self, data):
        pass

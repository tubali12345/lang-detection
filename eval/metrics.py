import torch

from config import Config
from evaluate import get_lang_by_id


class Metrics:
    def __init__(self,
                 predicted: torch.Tensor,
                 labels: torch.Tensor):
        assert predicted.shape == labels.shape, f'pred and labels tensor must have the same size, but got {predicted.shape} and {labels.shape}'
        self.predicted = predicted
        self.labels = labels

    def accuracy(self) -> float:
        return (torch.sum(self.predicted == self.labels) / self.labels.shape[0]).item()

    def acc_by_lang(self) -> dict:
        corr_by_lang = {lang: 0 for lang in Config.class_dict.keys()}
        no_by_lang = {lang: 0 for lang in Config.class_dict.keys()}
        for i in range(len(self.predicted)):
            no_by_lang[get_lang_by_id(self.labels[i])] += 1
            if self.predicted[i] == self.labels[i]:
                corr_by_lang[get_lang_by_id(self.labels[i])] += 1
        return {lang: corr_by_lang[lang] / no_by_lang[lang] for lang in Config.class_dict.keys()}

from datetime import date
from pathlib import Path

import torch

from config import Config


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


def get_lang_by_id(lang_id: int) -> str:
    return list(Config.class_dict.keys())[list(Config.class_dict.values()).index(lang_id)]


def print_and_write_metrics(accuracy: float,
                            acc_by_lang: float,
                            model_name: str,
                            file_path: str = f'/home/turib/lang_detection/eval/scores_{date.today()}.txt',
                            print_only: bool = False):
    print(f'Model: {model_name} \n'
          f'Accuracy: {accuracy} \n'
          f'Accuracy by language: {acc_by_lang}')
    if not print_only:
        Path(file_path).open('a').write(f'{model_name} \n'
                                        f'Accuracy: {accuracy}'
                                        f' \n Accuracy by language: '
                                        f' \n {acc_by_lang} \n')

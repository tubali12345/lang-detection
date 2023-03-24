from datetime import date
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from data.audio_preprocess import Aud2Mel, load_audio, split_audio
from data.dataloader import ShortAudioDataSet, collate_fn
from metrics import Metrics, get_lang_by_id, print_and_write_metrics
from models import call_whisper
from models.resnet import ResNet50LangDetection
from whisper.whisper import load_model


class EvaluateModel:
    def __init__(self,
                 model,
                 model_name: str = None,
                 weights_path: str = None,
                 device: str = 'cuda:0'):
        if model == "whisper":
            self.model = load_model('large', device=device)
            self.model_name = "whisper"
        else:
            self.model = model
            self.model_name = model_name
            self.model.to(device)
            self.model.load_state_dict(torch.load(weights_path))
            self.model.eval()
            self.aud_to_mel = Aud2Mel(Config.feature_dim, Config.sample_rate, 2048, 400, 160)
        self.device = device

    def evaluate_short(self, test_data_path: str, no_samples_per_class: int = 10 ** 20) -> tuple or str:
        pred = torch.Tensor([]).to(self.device)
        labels = torch.Tensor([]).to(self.device)

        if self.model_name == "whisper":
            return 'For model whisper use method evaluate_long'

        with torch.no_grad():
            for batch in tqdm(load_data(test_data_path, no_samples_per_class=no_samples_per_class), desc='Validating'):
                x, y = batch

                mel = self.aud_to_mel(x)
                out = self.model(mel.transpose(1, 2).to(self.device))
                _, batch_predicted = out.max(1)

                pred = torch.cat((pred, batch_predicted), dim=0)
                labels = torch.cat((labels, y.to(self.device)), dim=0)

        metrics = Metrics(pred, labels)
        return metrics.accuracy(), metrics.acc_by_lang()

    def evaluate_long(self,
                      test_data_path,
                      batch_audio_chunk: int = 32,
                      audio_format: str = 'wav',
                      no_samples_per_class: int = 10 ** 20) -> tuple:
        predicted = torch.Tensor([]).to(self.device)
        labels = torch.Tensor([]).to(self.device)

        with torch.no_grad():
            for language in Path(test_data_path).glob('*/'):
                for aud_path in tqdm(list(map(str, Path(language).rglob(f'*.{audio_format}')))[:no_samples_per_class]):
                    audio_class = aud_path.split('/')[-3]
                    if self.model_name == "whisper":
                        predicted_class = call_whisper.detect_language(model=self.model, audio_path=aud_path)
                    else:
                        predicted_class = self.predict_from_file(aud_path, batch_audio_chunk)
                    if predicted_class is None:
                        continue
                    try:
                        predicted = torch.cat(
                            (predicted, torch.Tensor([Config.class_dict[predicted_class]]).to(self.device)), dim=0)
                    except KeyError:  # whisper predicts language that is not in our languages
                        predicted = torch.cat(
                            (predicted, torch.Tensor([6]).to(self.device)), dim=0)
                    labels = torch.cat((labels, torch.Tensor([Config.class_dict[audio_class]]).to(self.device)), dim=0)

            metrics = Metrics(predicted, labels)
            return metrics.accuracy(), metrics.acc_by_lang()

    def predict_from_file(self, audio_path: str, batch_audio_chunk: int) -> str or None:
        audio = load_audio(audio_path)
        try:
            audio_chunks = torch.cat([torch.unsqueeze(torch.FloatTensor(aud), dim=0) for aud in split_audio(audio)])
        except Exception as e:
            print(f'Incorrect audio {audio_path}, exception: {e}')
            return

        predicted = torch.Tensor([]).to(self.device)

        with torch.no_grad():
            for i in range(0, len(audio_chunks), batch_audio_chunk):
                mel = self.aud_to_mel(audio_chunks[i:i + batch_audio_chunk])
                out = self.model(mel.transpose(1, 2).to(self.device))
                _, batch_predicted = out.max(1)
                predicted = torch.cat((predicted, batch_predicted), dim=0)

        return get_lang_by_id(int(np.argmax(np.bincount(list(map(int, predicted))))))


def load_data(data_path: str,
              no_samples_per_class: int,
              batch_size: int = 64,
              prefetch_factor: int = 4,
              num_workers: int = 1,
              ) -> DataLoader:
    return DataLoader(ShortAudioDataSet(data_path, random_sample=False, no_samples_per_class=no_samples_per_class),
                      batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                      pin_memory=False, num_workers=num_workers, prefetch_factor=prefetch_factor)


if __name__ == '__main__':
    # for i in [8, 7]:
    #     eval_resnet50 = EvaluateModel(ResNet50LangDetection(num_classes=len(Config.class_dict)),
    #                                   weights_path=f'/home/turib/lang_detection/weights/ResNet50/weights_03_08/model_{i}.pth',
    #                                   device='cuda:2')
    #     acc, acc_lang = eval_resnet50.evaluate_long(test_data_path='/home/turib/test_data_long',
    #                                                 no_samples_per_class=1000)
    #     print_and_write_metrics(acc, acc_lang, f'ResNet50 epoch-{i}')
    #     eval_resnet50.evaluate_short('/home/turib/val_data')

    eval_whisper = EvaluateModel(model='whisper', device='cuda:2')
    acc, acc_lang = eval_whisper.evaluate_long(test_data_path='/home/turib/test_data_long', no_samples_per_class=10)
    print_and_write_metrics(acc, acc_lang, eval_whisper.model_name, print_only=True)

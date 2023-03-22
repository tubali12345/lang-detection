from datetime import date
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from data.audio_preprocess import Aud2Mel, load_audio, split_audio
from data.dataloader import ShortAudioDataSet, collate_fn
from metrics import Metrics, get_lang_by_id
from models.resnet import ResNet50LangDetection


class EvaluateModel:
    def __init__(self,
                 model,
                 weights_path: str,
                 device: str):
        self.model = model
        self.model.to(device)
        self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()
        self.aud_to_mel = Aud2Mel(Config.feature_dim, Config.sample_rate, 2048, 400, 160)
        self.device = device

    def evaluate_short(self, test_data_path: str, no_samples_per_class: int = 10 ** 20) -> None:
        t_pred = torch.Tensor([]).to(self.device)
        t_labels = torch.Tensor([]).to(self.device)

        with torch.no_grad():
            for batch in tqdm(load_data(test_data_path, no_samples_per_class=no_samples_per_class), desc='Validating'):
                x, y = batch

                mel = self.aud_to_mel(x)
                out = self.model(mel.transpose(1, 2).to(self.device))
                _, batch_predicted = out.max(1)

                t_pred = torch.cat((t_pred, batch_predicted), 0)
                t_labels = torch.cat((t_labels, y.to(self.device)), 0)

        m2 = Metrics(t_pred, t_labels)
        print(m2.accuracy())
        print(m2.acc_by_lang())

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
                    predicted_class = self.predict_from_file(aud_path, batch_audio_chunk)
                    if predicted_class is None:
                        print('Incorrect audio')
                    predicted = torch.cat(
                        (predicted, torch.Tensor([Config.class_dict[predicted_class]]).to(self.device)), dim=0)
                    labels = torch.cat((labels, torch.Tensor([Config.class_dict[audio_class]]).to(self.device)), dim=0)

            m = Metrics(predicted, labels)
            print(m.acc_by_lang())
            print(m.accuracy())
            return m.accuracy(), m.acc_by_lang()

    def predict_from_file(self, audio_path: str, batch_audio_chunk: int) -> str or None:
        audio = load_audio(audio_path)
        try:
            audio_chunks = torch.cat([torch.unsqueeze(torch.FloatTensor(aud), 0) for aud in split_audio(audio)])
        except Exception as e:
            print(audio_path, e)
            return

        predicted = torch.Tensor([]).to(self.device)

        with torch.no_grad():
            for i in range(0, len(audio_chunks), batch_audio_chunk):
                mel = self.aud_to_mel(audio_chunks[i:i + batch_audio_chunk])
                out = self.model(mel.transpose(1, 2).to(self.device))
                _, batch_predicted = out.max(1)
                predicted = torch.cat((predicted, batch_predicted), 0)

        return get_lang_by_id(int(np.argmax(np.bincount(list(map(int, predicted))))))


def load_data(data_path: str,
              no_samples_per_class,
              batch_size: int = 64,
              prefetch_factor: int = 4,
              num_workers: int = 1,
              ) -> DataLoader:
    return DataLoader(ShortAudioDataSet(data_path, random_sample=False, no_samples_per_class=no_samples_per_class),
                      batch_size=batch_size, shuffle=False,
                      collate_fn=collate_fn, pin_memory=False, num_workers=num_workers, prefetch_factor=prefetch_factor)


if __name__ == '__main__':
    for i in [7]:
        eval_resnet50 = EvaluateModel(ResNet50LangDetection(num_classes=len(Config.class_dict)),
                                      weights_path=f'/home/turib/lang_detection/weights/ResNet50/weights_03_08/model_{i}.pth',
                                      device='cuda:0')
        acc, acc_by_lang = eval_resnet50.evaluate_long(test_data_path='/home/turib/test_data_long')
        Path(f'/home/turib/lang_detection/eval/scores_{date.today()}.txt').open('a').write(f'Epoch: {i} \n'
                                                                                           f'Accuracy: {acc}'
                                                                                           f' \n Accuracy by language: '
                                                                                           f' \n {acc_by_lang} \n')
        # eval_resnet50.evaluate_short('/home/turib/val_data')

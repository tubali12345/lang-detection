import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader

from config import Config
from data.audio_preprocess import Aud2Mel
from data.dataloader import ShortAudioDataSet, collate_fn
from models.resnet import ResNet50LangDetection
from train_loop import train_loop


# class TrainModel:
#     def __init__(self,
#                  model,
#                  train_ds: DataLoader,
#                  valid_ds: DataLoader,
#                  num_epochs: int,
#                  load_epoch: int,
#                  lr: float,
#                  max_lr: float,
#                  pci_start: int = 0.1,
#                  weights_path: str = None,
#                  out_dir_path: str = '',
#                  device: str = 'cuda:0'):
#         self.model = model
#         self.model.to(device)
#         if weights_path is not None:
#             self.model.load_state_dict(torch.load(weights_path))
#         self.aud_to_mel = Aud2Mel(Config.feature_dim, Config.sample_rate, 2048, 400, 160)
#
#         self.train_ds = train_ds
#         self.valid_ds = valid_ds
#
#         self.num_epochs = num_epochs
#         self.load_epoch = load_epoch
#         self.lr = lr
#         self.max_lr = max_lr
#         self.pci_start = pci_start
#
#         self.out_dir = _make_dir(out_dir_path)
#         self.device = device
#
#         self.optimizer = optimizer
#         self.loss_fn = loss_fn
#         self.lr_sched = lr_sched
#
#     def _train_loop(self):
#         for epoch in range(self.load_epoch + 1, self.num_epochs + 1):
#             self.model.train()
#             for i, batch in enumerate(tqdm(self.train_ds, desc=f'Training... Epoch {epoch}')):
#                 x, y = batch
#                 mel = self.aud_to_mel(x)
#                 out = self.model(mel.transpose(1, 2).to(self.device))
#                 loss = self.loss_fn(out, y.to(self.device))
#                 loss.backward()
#                 self.optimizer.step()
#                 self.optimizer.zero_grad()
#                 self.lr_sched.step()
#                 if i % 500 == 0:
#                     print(f'Current loss: {round(loss.item(), 4)}, current LR: {round(self.lr_sched.get_last_lr()[0], 6)}')
#             validation(self.model, self.valid_ds, self.loss_fn, self.aud_to_mel, self.out_dir, round(self.lr_sched.get_last_lr()[0], 6), self.device)
#             torch.save(self.model.state_dict(), f'{self.out_dir}/model_{epoch}.pth')
#
#     def val_loop(self):
#         pass
#
#


def load_data(train_data_path: str,
              val_data_path: str,
              batch_size: int,
              prefetch_factor: int = 4,
              num_workers: int = 1
              ) -> tuple:
    ds = DataLoader(ShortAudioDataSet(train_data_path, with_augmentation=True), batch_size=batch_size, shuffle=True,
                    collate_fn=collate_fn, drop_last=False, pin_memory=False, num_workers=num_workers,
                    prefetch_factor=prefetch_factor)
    valid_ds = DataLoader(ShortAudioDataSet(val_data_path, random_sample=False, no_samples_per_class=10**5),
                          batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=False,
                          num_workers=num_workers, prefetch_factor=prefetch_factor)
    return ds, valid_ds


# add option to train model from a certain weights
# add tensorboard
def train(num_epochs: int,
          lr: float,
          max_lr: float,
          batch_size: int,
          num_classes: int,
          out_dir_path: str,
          pct_start: float = 0.1,
          load_epoch: int = 0,
          weights_path: str = None,
          device: str = 'cuda:0'):
    ds, valid_ds = load_data(train_data_path='/home/turib/train_data',
                             val_data_path='/home/turib/train_data_val',
                             batch_size=batch_size)

    model = ResNet50LangDetection(num_classes=num_classes).to(device)

    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(reduction='mean')

    div_factor = max_lr / 3e-6
    lr_sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, num_epochs * len(ds),
                                                   div_factor=div_factor,
                                                   pct_start=pct_start,
                                                   final_div_factor=div_factor)
    train_loop(model=model,
               num_epochs=num_epochs,
               train_ds=ds,
               valid_ds=valid_ds,
               aud_to_mel=Aud2Mel(Config.feature_dim, Config.sample_rate, 2048, 400, 160),
               loss_fn=loss_fn,
               optimizer=optimizer,
               lr_sched=lr_sched,
               out_dir_path=out_dir_path,
               load_epoch=load_epoch,
               device=device)


if __name__ == '__main__':
    train(num_epochs=100,
          lr=1e-4,
          max_lr=3e-4,
          batch_size=64,
          num_classes=6,
          out_dir_path=f'/home/turib/lang_detection/weights/ResNet50/weights_03_08',
          device='cuda:0')

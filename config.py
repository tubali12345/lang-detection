# change to yaml
class Config:
    feature_dim = 80
    sample_rate = 16000
    class_dict = {'tr': 0, 'fr': 1, 'de': 2, 'hu': 3, 'en': 4, 'es': 5}
    train_params = {'train_data_path': '/home/turib/train_data',
                    'val_data_path': '/home/turib/train_data_val',
                    'no_samples_per_class_train': 5.4 * 10 ** 5,
                    'no_samples_per_class_val': 10 ** 5,
                    'num_epochs': 100,
                    'lr': 1e-4,
                    'max_lr': 3e-4,
                    'pct_start': 0.1,
                    'batch_size': 64,
                    'num_classes': len(class_dict),
                    'out_dir_path': '/home/turib/lang_detection/weights/ResNet50/weights',
                    'weights_path': None,
                    'load_epoch': 0,
                    'device': 'cuda:0'}

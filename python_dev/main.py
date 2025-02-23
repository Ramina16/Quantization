import torch
import argparse
import json

from data_loader import CustomDataLoader
from main_functions import evaluate, load_model
from constants import KEY_ACCURACY, KEY_NUM_IMAGES, KEY_RECALL, KEY_PRECISION, KEY_NUM_CLASSES , KEY_PATH_TO_DATA
from training import Trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training and evaluation')
    parser.add_argument('--config_path', '-config_path', type=str, help='Path to config file with model and training parameters',
                        default='default_config.json')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(args.config_path, 'r') as f:
        config = json.loads(f.read())

    config['use_pretrained_w'] = True if config['use_pretrained_w'] == 'True' else False

    loader = CustomDataLoader(path_to_data=config[KEY_PATH_TO_DATA])
    train_loader, val_loader, test_loader = loader.get_train_test_data()

    trainer = Trainer(train_loader, val_loader, test_loader, device, config)

    trainer.train()

    # Evaluate best saved model on the whole test dataset
    model = load_model(device, use_pretrained_w=False, num_classes=config[KEY_NUM_CLASSES])
    test_metrics = evaluate(model, test_loader, num_classes=config[KEY_NUM_CLASSES], device=device)

    print(f'Accuracy is {test_metrics[KEY_ACCURACY]:.4f}, Recall is {test_metrics[KEY_RECALL]:.4f}, Precision is {test_metrics[KEY_PRECISION]:.4f} \
          for {test_metrics[KEY_NUM_IMAGES]} images among {len(test_loader) * test_loader.batch_size} images')

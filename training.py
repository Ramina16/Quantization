import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from data_loader import CustomDataLoader
from main_functions import evaluate, load_model, train
from vgg_13 import VGG_13
from callbacks_utils import SaveBestCheckpointCallback


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pretrained = True
    model = VGG_13(num_classes=10)
    model = model.to(device)
    if pretrained is not None:
        weights = '/mnt/Volume_HDD_8TB_1/olena_models/vgg13_bn-abd245e5.pth'
        state_dict = torch.load(weights, map_location=torch.device(device))
        filtered_state_dict = {k: v for k, v in state_dict.items() if "classifier" not in k}
        model.load_state_dict(filtered_state_dict, strict=False)
        model.classifier = model._create_classifier()
        model.classifier = model.classifier.to(device)
        for param in model.features.parameters():
            param.requires_grad = False
    

    loader = CustomDataLoader(path_to_data='/mnt/Volume_HDD_8TB_1/olena_datasets/dataset_animals/raw-img')
    train_loader, val_loader, test_loader = loader.get_train_test_data()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-5)

    train(model, train_loader, val_loader, criterion, optimizer, callbacks=[SaveBestCheckpointCallback('best_model.pth')], device=device)
    acc = evaluate(model, test_loader, device=device, stop=False)


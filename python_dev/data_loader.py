import numpy as np
import random

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from constants import IMAGE_NET_MEAN, IMAGE_NET_STD


translate_classes = {"cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly", "gallina": "chicken", 
                     "gatto": "cat", "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel", "dog": "cane", "cavallo": "horse", 
                     "elephant" : "elefante", "butterfly": "farfalla", "chicken": "gallina", "cat": "gatto", "cow": "mucca", "spider": "ragno", 
                     "squirrel": "scoiattolo"}


seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


class CustomDataLoader():
    """
    
    """
    def __init__(self, path_to_data: str, batch_size: int = 32, train_split: float = 0.7, val_split: float = 0.2):
        """
        
        """
        self.path_to_data = path_to_data
        self.full_data = datasets.ImageFolder(self.path_to_data)

        self.batch_size = batch_size
        self.train_split = train_split
        self.val_split = val_split
    
    def preprocess_datalabels(self):
        for i in range(len(self.full_data.classes)):
            label = self.full_data.classes[i]
            if label in translate_classes:
                self.full_data.classes[i] = translate_classes[label]

    
    def preprocess_data(self, train: bool):
        """
        
        """
        if train:
            transforms_data = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=20),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
            ])
        else:
            transforms_data = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
            ])
        
        return transforms_data

    def get_train_test_data(self, ):
        """
        
        """
        train_size = int(len(self.full_data) * self.train_split)
        val_size = int(len(self.full_data) * self.val_split)
        test_size = len(self.full_data) - train_size - val_size
        train_data, val_data, test_data = random_split(self.full_data, [train_size, val_size, test_size])

        train_data.dataset.transform = self.preprocess_data(train=True)
        val_data.dataset.transform = self.preprocess_data(train=True)
        test_data.dataset.transform = self.preprocess_data(train=False)

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

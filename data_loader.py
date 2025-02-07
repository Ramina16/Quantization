from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
import random

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


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
    def __init__(self, path_to_data: str, train: bool = True):
        """
        
        """
        self.path_to_data = path_to_data
        self.full_data = datasets.ImageFolder(self.path_to_data)
        for i in range(len(self.full_data.classes)):
            label = self.full_data.classes[i]
            if label in translate_classes:
                self.full_data.classes[i] = translate_classes[label]
        print(len(self.full_data.classes))

    
    def preprocess_data(self, train: bool):
        """
        
        """
        if train:
            transforms_data = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=20),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transforms_data = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        return transforms_data

    def get_train_test_data(self, train_split: float = 0.7, val_split: float = 0.2, test_split: float = 0.1, batch_size=32):
        """
        
        """
        train_size = int(len(self.full_data) * train_split)
        val_size = int(len(self.full_data) * val_split)
        test_size = len(self.full_data) - train_size - val_size
        train_data, val_data, test_data = random_split(self.full_data, [train_size, val_size, test_size])

        train_data.dataset.transform = self.preprocess_data(train=True)
        val_data.dataset.transform = self.preprocess_data(train=True)
        test_data.dataset.transform = self.preprocess_data(train=False)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader


# loader = CustomDataLoader(path_to_data='dataset_animals/raw-img')
# train_loader, val_loader, test_loader = loader.get_train_test_data()

# data_iter = iter(test_loader)
# images, labels = next(data_iter)

# print(f"Batch size: {images.shape[0]}, Image shape: {images.shape[1:]}")

# image, label = images[0], labels[0]

# # Convert tensor image to NumPy format
# image = image.permute(1, 2, 0).numpy()  # Change shape from (C, H, W) → (H, W, C)

# # Undo normalization if needed
# mean=[0.485, 0.456, 0.406]
# std=[0.229, 0.224, 0.225]
# image = image * std + mean  # Denormalize

# # Display the image
# plt.imshow(image)
# plt.title(f"Label: {test_loader.dataset.dataset.classes[label]}")
# plt.axis("off")
# plt.show()
    

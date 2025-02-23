import torch
import torch.nn as nn
from tqdm import tqdm

from main_functions import evaluate, load_model
from constants import (KEY_ACCURACY, KEY_LR, KEY_NAME_BEST_MODEL, KEY_PRECISION, KEY_RECALL, MIN_EPOCHS_B_TEST, KEY_BATCH_SIZE, KEY_MAX_TEST_IMG, KEY_NUM_CLASSES,
                       KEY_NUM_EPOCHS, KEY_OPTIMIZER, KEY_USE_PRETRAINED_W)
from callbacks_utils import SaveBestCheckpointCallback


class Trainer:
    """
    
    """
    def __init__(self, train_dataloader, val_dataloader, test_dataloader, device, config):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.device = device

        self.config = config

        # load model
        self.model = load_model(device=self.device, use_pretrained_w=self.config[KEY_USE_PRETRAINED_W], num_classes=config[KEY_NUM_CLASSES])
        # set optimizer
        self.set_optimizer()
        # set loss
        self.criterion = nn.CrossEntropyLoss()
        # set callbacks
        self.prepare_callbacks()

        self.best_val_loss = float('inf')
        self.test_epoch = 0
    
    def set_optimizer(self, ):
        if self.config[KEY_OPTIMIZER] == 'adam':
            self.optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=self.config[KEY_LR])
        elif self.config[KEY_OPTIMIZER] == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.classifier.parameters(), lr=self.config[KEY_LR])

    def prepare_callbacks(self, ):
        self.callbacks = [SaveBestCheckpointCallback(KEY_NAME_BEST_MODEL), ]

    def train_step(self, epoch):
        self.model.train()
        train_loss = 0.
        train_acc = 0.

        for inputs, targets in tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1} / {self.config[KEY_NUM_EPOCHS]} [Train]", leave=False):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

            outputs = torch.argmax(outputs, dim=1)
            train_acc += torch.sum(outputs == targets)
        
        train_loss /= (len(self.train_dataloader) * self.config[KEY_BATCH_SIZE])
        train_acc /= (len(self.train_dataloader) * self.config[KEY_BATCH_SIZE])

        return train_loss, train_acc

    def eval_step(self, epoch):
        self.model.eval()
        val_loss = 0.
        val_acc = 0.
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_dataloader, desc=f"Epoch {epoch + 1} / {self.config[KEY_NUM_EPOCHS]} [Val]", leave=False):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()

                outputs = torch.argmax(outputs, dim=1)
                val_acc += torch.sum(outputs == targets)

        val_loss /= (len(self.val_dataloader) * self.config[KEY_BATCH_SIZE])
        val_acc /= (len(self.val_dataloader) * self.config[KEY_BATCH_SIZE])

        return val_loss, val_acc

    def test_step(self, val_loss, epoch):
        self.best_val_loss = val_loss
        self.test_epoch = epoch
        test_metrics = evaluate(self.model, self.test_dataloader, num_classes=self.config[KEY_NUM_CLASSES], device=self.device, 
                                max_img=self.config[KEY_MAX_TEST_IMG])

        return test_metrics
    
    def train(self, ):
        for epoch in range(self.config[KEY_NUM_EPOCHS]):
            train_loss, train_acc = self.train_step(epoch)
            val_loss, val_acc = self.eval_step(epoch)
            
            print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")

            if val_loss < self.best_val_loss and (epoch - self.test_epoch) >= MIN_EPOCHS_B_TEST:
                test_metrics = self.test_step(val_loss, epoch)

                print(f"Epoch {epoch + 1}: Test accuracy = {test_metrics[KEY_ACCURACY]:.4f}, \
                      Test recall = {test_metrics[KEY_RECALL]:.4f}, \
                      Test precision = {test_metrics[KEY_PRECISION]:.4f}")


            if self.callbacks:
                for callback in self.callbacks:
                    callback(self.model, val_loss)

import torch
import os

from constants import CHECKPOINTS_FOLDER


class SaveBestCheckpointCallback:
    def __init__(self, save_path: str):
        self.best_val_loss = float('inf')
        os.makedirs(CHECKPOINTS_FOLDER, exist_ok=True)
        self.save_path = os.path.join(CHECKPOINTS_FOLDER, save_path)
    
    def __call__(self, model, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(model.state_dict(), self.save_path)
            print(f"New best model saved at {self.save_path} (Val Loss: {val_loss:.4f})")

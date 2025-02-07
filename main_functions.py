import torch
import os

from tqdm import tqdm
from vgg_13 import VGG_13
from callbacks_utils import CHECKPOINTS_FOLDER


def load_model(model=None, device='cpu'):
    if model is None:
        model = VGG_13(num_classes=10)
        model = model.to(device)

    pretrained_model = os.path.join(CHECKPOINTS_FOLDER, 'best_model.pth')
    if os.path.exists(pretrained_model):
        state_dict = torch.load(pretrained_model, map_location=torch.device(device))
        model.load_state_dict(state_dict)
        print(f'Successfully loaded existed checkpoint {pretrained_model} into the model')
    
    return model


def train(model, train_dataloader, val_dataloader, criterion, optimizer, callbacks, num_epochs: int = 10, batch_size=32, device='cpu'):
    """
    
    """
    model = load_model(model)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.
        train_acc = 0.

        for inputs, targets in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            #scheduler.step()

            train_loss += loss.item()

            outputs = torch.argmax(outputs, dim=1)
            train_acc += torch.sum(outputs == targets)
            
        train_loss /= (len(train_dataloader) * batch_size)
        train_acc /= (len(train_dataloader) * batch_size)

        model.eval()
        val_loss = 0.
        val_acc = 0.
        with torch.no_grad():
            for inputs, targets in tqdm(val_dataloader, desc=f"Epoch {epoch + 1} / {num_epochs} [Val]", leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                outputs = torch.argmax(outputs, dim=1)
                val_acc += torch.sum(outputs == targets)

        val_loss /= (len(val_dataloader) * batch_size)
        val_acc /= (len(val_dataloader) * batch_size)

        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")

        if callbacks:
            for callback in callbacks:
                callback(model, val_loss)
    

def evaluate(model, dataloader, batch_size=32, device='cpu', q=False, max_img=None):
    """
    
    """
    model.eval()
    accuracy = 0
    count = 0
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader):
            if max_img is not None and count > max_img:
                break

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if q:
                outputs = outputs.int_repr()
            outputs = torch.argmax(outputs, dim=1)

            accuracy += torch.sum(outputs == targets).item()

            count += batch_size

    accuracy /= count

    print(f'Accuracy is {accuracy:.4f}')

    return accuracy


def calibrate_model(model, dataloader, num_images=100):
    model.eval()

    count = 0
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader):
            if count > num_images:
                break

            outputs = model(inputs)
            count += len(inputs)

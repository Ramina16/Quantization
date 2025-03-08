import torch
import os

from tqdm import tqdm
from vgg_13 import QVGG_13, VGG_13
from callbacks_utils import CHECKPOINTS_FOLDER

from constants import KEY_ACCURACY, KEY_NAME_BEST_MODEL, KEY_NUM_IMAGES, KEY_PRECISION, KEY_RECALL


def load_model(device='cpu', use_pretrained_w=True, num_classes=10, quantized=False):
    """
    
    """
    model = VGG_13(num_classes=num_classes) if quantized else QVGG_13(num_classes=num_classes)
    model = model.to(device)
    if use_pretrained_w:
        pretrained_w = os.path.join('pretrained_weights', 'vgg13_bn-abd245e5.pth')
        state_dict = torch.load(pretrained_w, map_location=torch.device(device))
        filtered_state_dict = {k: v for k, v in state_dict.items() if "classifier" not in k}
        model.load_state_dict(filtered_state_dict, strict=False)
        model.classifier = model._create_classifier()
        model.classifier = model.classifier.to(device)

        for param in model.features.parameters():
            param.requires_grad = False
            
        print(f'Successfully loaded pretrained weights on ImageNet {pretrained_w} into the model')

    else:
        pretrained_model = os.path.join(CHECKPOINTS_FOLDER, KEY_NAME_BEST_MODEL)
        if os.path.exists(pretrained_model):
            state_dict = torch.load(pretrained_model, map_location=torch.device(device))
            model.load_state_dict(state_dict)
            for param in model.features.parameters():
                param.requires_grad = False

            print(f'Successfully loaded existed checkpoint {pretrained_model} into the model')
    
    return model
    

def evaluate(model, dataloader, num_classes=10, device='cpu', max_img=None):
    """
    
    """
    model.eval()

    outputs_all, targets_all = [], []

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader):
            if max_img is not None and count > max_img:
                break

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            outputs_all.append(outputs)
            targets_all.append(targets)

    outputs_all = torch.cat(outputs_all)
    preds_all = torch.argmax(outputs_all, dim=1)
    targets_all = torch.cat(targets_all)

    count = targets_all.shape[0]

    accuracy = (preds_all == targets_all).sum().item() / count

    precision_per_class = []
    recall_per_class = []

    for c in range(num_classes):
        TP = ((preds_all == c) & (targets_all == c)).sum().item()
        FP = ((preds_all == c) & (targets_all != c)).sum().item()
        FN = ((preds_all != c) & (targets_all == c)).sum().item()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    precision_per_class.append(precision)
    recall_per_class.append(recall)

    precision_macro = sum(precision_per_class) / num_classes
    recall_macro = sum(recall_per_class) / num_classes

    metrics = {KEY_ACCURACY: accuracy,
               KEY_RECALL: recall_macro,
               KEY_PRECISION: precision_macro,
               KEY_NUM_IMAGES: count}

    return metrics


def calibrate_model(model, dataloader, num_images=100):
    model.eval()

    count = 0
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader):
            if count > num_images:
                break

            outputs = model(inputs)
            count += dataloader.batch_size
    
    return model

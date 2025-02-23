import torch
import onnxruntime
import cv2
import numpy as np

from main_functions import load_model
from vgg_13 import QVGG_13

session_fp32 = onnxruntime.InferenceSession("checkpoints/q_best_model_per_tensor_d.onnx")
#session_fp32 = onnxruntime.InferenceSession("/Users/o_kolodiazhna/Downloads/projects/python_projects/quantization_android/android_classififcation/app/src/main/res/raw/mobilenetv2_fp32.onnx")

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.transpose([2, 0, 1])

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    img = img.astype(np.float32)

    for channel in range(img.shape[0]):
        img[channel, :, :] = (img[channel, :, :] / 255. - mean[channel]) / std[channel]
    img = np.expand_dims(img, 0)

    return img

def run_sample(session, image_file, categories):
    output = session.run([], {'input':preprocess_image(image_file)})[0]
    output = output.flatten()
    output = softmax(output) # this is optional
    top5_catid = np.argsort(-output)[:5]
    for catid in top5_catid:
        print(categories[catid], output[catid])

with open("classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

run_sample(session_fp32, 'dataset_animals/raw-img/cane/OIP-_5Em--O1RA44HxiWK_ybawHaF4.jpeg', categories)


# def run_sample(session, image_file):
#     output = session.run([], {'input': preprocess_image(image_file)})[0]
#     output = np.argmax(output, 1)
#     print(output)

# categories = []
#run_sample(session_fp32, 'dataset_animals/raw-img/mucca/OIP-__wtYf6sbvlOlMBMHyA3OAAAAA.jpeg')

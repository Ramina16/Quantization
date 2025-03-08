# Quantization
## This project trains and evaluates a deep learning image classification model, then converts and quantizes it using ONNX for mobile deployment. An Android Studio app integrates the quantized model, enabling real-time video and gallery-based image classification.

## **Example Usage:**
### Model Training and Evaluation:
```bash
python3 -m main --config_path 'python_dev/default_config.json'
```
You can find all training parameters as well as data path in *python_dev/default_config.json*.

### Model quantization:
```bash
python3 -m model_quantization --config_path 'python_dev/default_config.json'
```
This script will save quantized (to INT8) .onnx model. Outputs are dequantized to FLOAT32, but you can remove dequantized layer in *python_dev/vgg_13.py* to store outputs in INT8 format.

### Start an adroid application:
```bash
You need to install Android Studio and сorresponding SDKs. Then run *android* folder as project in android srudio, build it and run. You should connect your mobile phone to your PC/laptop. 
```

<img width="250" alt="image" src="https://github.com/user-attachments/assets/dfea01ce-3727-4698-a428-8a3b2666417b" />
<img width="250" alt="image" src="https://github.com/user-attachments/assets/3a001aa9-363e-4bf4-ae47-7d7a7420c01f" />
<img width="250" alt="image" src="https://github.com/user-attachments/assets/c13e45fd-d27f-48f1-afcb-cd924a5065fa" />

![til](videos/github.com/Ramina16/Quantization/tree/main/assets/demo.gif)





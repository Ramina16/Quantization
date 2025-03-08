import torch

from data_loader import CustomDataLoader
from main_functions import calibrate_model, evaluate, load_model
from python_dev.constants import KEY_ACCURACY, KEY_NUM_IMAGES, KEY_PRECISION, KEY_RECALL
from vgg_13 import QVGG_13



if __name__ == '__main__':
    loader = CustomDataLoader(path_to_data='dataset_animals/raw-img')
    train_loader, val_loader, test_loader = loader.get_train_test_data()

    # quantize model to int8
    qmodel = load_model(use_pretrained_w=False, quantized=True)
    qmodel.to('cpu')

    # fuse conv -> batchnorm -> relu
    qmodel.fuse_model()

    # per-channel
    qmodel.qconfig = torch.ao.quantization.get_default_qconfig('x86')
    # per tensor
    #qmodel.qconfig = torch.ao.quantization.default_qconfig

    torch.ao.quantization.prepare(qmodel, inplace=True)

    acc = calibrate_model(qmodel, train_loader)

    torch.ao.quantization.convert(qmodel, inplace=True)

    metrics_qmodel = evaluate(qmodel, test_loader)
    print(f'Metrics for quantized model for {metrics_qmodel[KEY_NUM_IMAGES]} images:\n \
            accuracy: {metrics_qmodel[KEY_ACCURACY]}\n, \
            recall: {metrics_qmodel[KEY_RECALL]},\n \
            precision: {metrics_qmodel[KEY_PRECISION]}: precision_macro')

    torch.jit.save(torch.jit.script(qmodel), 'checkpoints/q_best_model_per_ch_d.pth')

    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    torch_out = qmodel(x)

    # Export the model
    torch.onnx.export(qmodel,
                      x,
                      "checkpoints/q_best_model_per_ch_d.onnx",
                      export_params=True,
                      opset_version=12,
                      do_constant_folding=True,
                      input_names = ['input'],
                      output_names = ['output'])

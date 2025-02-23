import torch
torch.backends.quantized.engine = 'qnnpack'

from data_loader import CustomDataLoader
from main_functions import calibrate_model, evaluate, load_model
from vgg_13 import QVGG_13



if __name__ == '__main__':
    loader = CustomDataLoader(path_to_data='dataset_animals/raw-img')
    train_loader, val_loader, test_loader = loader.get_train_test_data()

    # evaluate float model
    # model = load_model(device='cpu')
    # model = model.eval()

    # model.to('cpu')
    # acc = evaluate(model, test_loader, device='cpu', max_img=150)

    # quantize model to int8
    qmodel = QVGG_13(num_classes=10)
    qmodel = load_model(qmodel)
    qmodel.to('cpu')

    qmodel.fuse_model()

    # per-channel
    qmodel.qconfig = torch.ao.quantization.get_default_qconfig('qnnpack')#('x86')
    # per tensor
    #qmodel.qconfig = torch.ao.quantization.default_qconfig

    torch.ao.quantization.prepare(qmodel, inplace=True)

    acc = calibrate_model(qmodel, train_loader)

    torch.ao.quantization.convert(qmodel, inplace=True)

    acc = evaluate(qmodel, test_loader, q=True, max_img=150)

    #torch.jit.save(torch.jit.script(qmodel), 'checkpoints/q_best_model_per_ch.pth')
    torch.jit.save(torch.jit.script(qmodel), 'checkpoints/q_best_model_per_ch_d.pth')

    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    torch_out = qmodel(x)

    # Export the model
    torch.onnx.export(qmodel,              # model being run
                      x,                         # model input (or a tuple for multiple inputs)
                      "checkpoints/q_best_model_per_ch_d.onnx", # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=12,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output']) # the model's output names
    

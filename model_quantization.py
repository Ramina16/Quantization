import torch

from data_loader import CustomDataLoader
from main_functions import calibrate_model, evaluate, load_model
from vgg_13 import QVGG_13



if __name__ == '__main__':
    loader = CustomDataLoader(path_to_data='/mnt/Volume_HDD_8TB_1/olena_datasets/dataset_animals/raw-img')
    train_loader, val_loader, test_loader = loader.get_train_test_data()

    # evaluate float model
    model = load_model(device='cuda')
    model = model.eval()

    model.to('cuda')
    acc = evaluate(model, test_loader, device='cuda')

    # quantize model to int8
    qmodel = QVGG_13(num_classes=10)
    qmodel = load_model(qmodel)
    qmodel.to('cpu')

    qmodel.fuse_model()

    # per-channel
    qmodel.qconfig = torch.ao.quantization.get_default_qconfig('x86')
    # per tensor
    #qmodel.qconfig = torch.ao.quantization.default_qconfig

    torch.ao.quantization.prepare(qmodel, inplace=True)

    acc = calibrate_model(qmodel, train_loader)

    torch.ao.quantization.convert(qmodel, inplace=True)

    acc = evaluate(qmodel, test_loader, q=True)

    #torch.jit.save(torch.jit.script(qmodel), 'checkpoints/q_best_model_per_ch.pth')
    

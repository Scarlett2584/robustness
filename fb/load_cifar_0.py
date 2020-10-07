import dill
import foolbox as fb
from foolbox import zoo
import torch

from robustness import model_utils
from robustness.datasets import CIFAR


def create():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weights = zoo.fetch_weights(weights_uri='https://www.dropbox.com/s/yhpp4yws7sgi6lj/cifar_nat.pt?dl=1',
                                unzip=False)
    ds = CIFAR('/tmp/')
    m = ds.get_model(arch='resnet50', pretrained=False)
    checkpt = torch.load(weights, pickle_module=dill, map_location=device)
    layer_keys = filter(lambda x: x.startswith('module.model'), checkpt['model'].keys())
    checkpt = {k[len('module.model.'):]: checkpt['model'][k] for k in layer_keys}
    m.load_state_dict(checkpt)
    m.eval()

    preprocessing = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010], axis=-3)

    fmodel = fb.models.PyTorchModel(m, bounds=(0, 1), preprocessing=preprocessing)

    return fmodel


if __name__ == '__main__':
    create()

import dill
import foolbox as fb
from foolbox import zoo
import torch

from robustness.datasets import CIFAR

robust_models = {'l2-0': 'https://www.dropbox.com/s/yhpp4yws7sgi6lj/cifar_nat.pt?dl=1',
                 'l2-0.25': 'https://www.dropbox.com/s/2qsp7pt6t7uo71w/cifar_l2_0_25.pt?dl=1',
                 'l2-0.5': 'https://www.dropbox.com/s/1zazwjfzee7c8i4/cifar_l2_0_5.pt?dl=1',
                 'l2-1.0': 'https://www.dropbox.com/s/s2x7thisiqxz095/cifar_l2_1_0.pt?dl=1',
                 'l-inf-0': 'https://www.dropbox.com/s/yhpp4yws7sgi6lj/cifar_nat.pt?dl=1',  # same as l2-0
                 'l-inf-8/255': 'https://www.dropbox.com/s/c9qlt1lbdnu9tlo/cifar_linf_8.pt?dl=1'}

model_url = robust_models['l2-0']
def create():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weights = zoo.fetch_weights(weights_uri=model_url,
                                unzip=False)
    ds = CIFAR('/tmp/')
    m = ds.get_model(arch='resnet50', pretrained=False)
    checkpt = torch.load(weights, pickle_module=dill, map_location=device)
    model_keys = ['model', 'state_dict']
    model_key = [k for k in model_keys if k in checkpt.keys()][0]
    layer_keys = filter(lambda x: x.startswith('module.model'), checkpt[model_key].keys())
    checkpt = {k[len('module.model.'):]: checkpt[model_key][k] for k in layer_keys}
    m.load_state_dict(checkpt)
    m.eval()

    preprocessing = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010], axis=-3)

    fmodel = fb.models.PyTorchModel(m, bounds=(0, 1), preprocessing=preprocessing)

    return fmodel


if __name__ == '__main__':
    m = create()

    import torchvision
    tr = torchvision.transforms.ToTensor()
    ds = torchvision.datasets.CIFAR10('~/.pytorch/CIFAR10', download=False, train=False, transform=tr)
    from torch.utils import data
    loader = torch.utils.data.DataLoader(ds, batch_size=100, num_workers=10)

    images, labels = next(iter(loader))
    acc = fb.accuracy(m, images, labels)
    print(acc)

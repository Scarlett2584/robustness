import dill
import foolbox as fb
import torch
import torchvision
from foolbox import zoo

from robustness.datasets import ImageNet

robust_models = {'l2-0': None,
                 'l2-3': 'https://www.dropbox.com/s/knf4uimlqsi1yz8/imagenet_l2_3_0.pt?dl=1',
                 'l-inf-0': None,  # same as l2-0
                 'l-inf-4/255': 'https://www.dropbox.com/s/axfuary2w1cnyrg/imagenet_linf_4.pt?dl=1',
                 'l-inf-8/255': 'https://www.dropbox.com/s/yxn15a9zklz3s8q/imagenet_linf_8.pt?dl=1'}


def create(model_id='l2-0'):
    model_url = robust_models[model_id]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_url is not None:
        weights = zoo.fetch_weights(weights_uri=model_url,
                                    unzip=False)
        ds = ImageNet('/tmp/')
        m = ds.get_model(arch='resnet50', pretrained=False)
        checkpt = torch.load(weights, pickle_module=dill, map_location=device)
        model_keys = ['model', 'state_dict']
        model_key = [k for k in model_keys if k in checkpt.keys()][0]
        layer_keys = filter(lambda x: x.startswith('module.model'), checkpt[model_key].keys())
        checkpt = {k[len('module.model.'):]: checkpt[model_key][k] for k in layer_keys}
        m.load_state_dict(checkpt)
    else:
        m = torchvision.models.resnet50(pretrained=True)
    m.eval()

    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)

    fmodel = fb.models.PyTorchModel(m, bounds=(0, 1), preprocessing=preprocessing)

    return fmodel

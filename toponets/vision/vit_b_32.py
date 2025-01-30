import torch
import os
import torchvision.models as models

from .resnet18 import BlurPoolConv2d, create_model_and_scaler

suffix = "?download=true"
base_url = "https://huggingface.co/murtylab/topo-vit-b-32-imagenet/resolve/main/"


possible_taus = [
    10.0
]

def get_name_from_tau(tau):
    tau = float(tau)
    assert tau in possible_taus, f"tau must be one of {possible_taus}"
    return f"tau_{tau}.pt"

def get_url_from_tau(tau):
    return base_url + get_name_from_tau(tau) + suffix

def vit_b_32(tau, checkpoint_path: str = None):
    if checkpoint_path is None: 
        checkpoint_path = f"vit_b_32_tau_{tau}.pt"

    model = models.vit_b_32(weights=None)

    url = get_url_from_tau(tau=tau)
    if not os.path.exists(checkpoint_path):
        os.system(
            f"wget -O {checkpoint_path} {url}"
        )
    else:
        pass

    state_dict = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(state_dict['model'])
    model.eval()
    return model

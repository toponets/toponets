import torch
import os

from .resnet18 import BlurPoolConv2d, create_model_and_scaler

suffix = "?download=true"
base_url = "https://huggingface.co/murtylab/topo-resnet50-imagenet/resolve/main/"

possible_taus = [
    30.0
]

def get_name_from_tau(tau):
    tau = float(tau)
    assert tau in possible_taus, f"tau must be one of {possible_taus}"
    return f"all_topo_tau_{tau}.pt"

def get_url_from_tau(tau):
    return base_url + get_name_from_tau(tau) + suffix

def resnet50(tau=30.0, checkpoint_path = None):

    if checkpoint_path is None: 
        checkpoint_path = f"resnet50_tau_{tau}.pt"

    model, scaler = create_model_and_scaler(
        arch="resnet50",
        pretrained=False,
        use_blurpool=True,
    )
    
    url = get_url_from_tau(tau=tau)
    if not os.path.exists(checkpoint_path):
        os.system(
            f"wget -O {checkpoint_path} {url}"
        )
    else:
        pass

    state_dict = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model
import torch
from torch.amp import GradScaler
import torchvision.models as models
import numpy as np
import os

class BlurPoolConv2d(torch.nn.Module):
    """
    Note to future Mayukh.

    some_blurpool_layer.conv is the same as some_blurpool_layer when it comes to extracting hook outputs
    look closely, the last thing in the forward pass is self.conv.

    No idea why the hook output is None for some_blurpool_layer.conv,
    but that is fine. Just set it to some_blurpool_layer and you'll get the equivalent result
    """
    def __init__(self, conv):
        super().__init__()
        default_filter = torch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
        filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        self.register_buffer('blur_filter', filt)

    def forward(self, x):
        blurred = F.conv2d(x, self.blur_filter, stride=1, padding=(1, 1),
                           groups=self.conv.in_channels, bias=None)
        return self.conv.forward(blurred)
    

suffix = "?download=true"
base_url = "https://huggingface.co/murtylab/topo-resnet18-imagenet/resolve/main/"

possible_taus = [
    0.5,
    1.0,
    5.0,
    10.0,
    20.0,
    50.0
]

def create_model_and_scaler(arch, pretrained, use_blurpool):
        scaler = GradScaler()
        model = getattr(models, arch)(pretrained=pretrained)
        def apply_blurpool(mod: torch.nn.Module):
            for (name, child) in mod.named_children():
                if isinstance(child, torch.nn.Conv2d) and (np.max(child.stride) > 1 and child.in_channels >= 16): 
                    setattr(mod, name, BlurPoolConv2d(child))
                else: apply_blurpool(child)
        if use_blurpool: apply_blurpool(model)

        model = model.to(memory_format=torch.channels_last)
        return model, scaler

def get_name_from_tau(tau):
    tau = float(tau)
    assert tau in possible_taus, f"tau must be one of {possible_taus}"
    return f"all_topo_tau_{tau}.pt"

def get_url_from_tau(tau):
    return base_url + get_name_from_tau(tau) + suffix

def resnet18(tau=0.5, checkpoint_path = None):

    if checkpoint_path is None: 
        checkpoint_path = f"resnet18_tau_{tau}.pt"

    model, scaler = create_model_and_scaler(
        arch="resnet18",
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
    return model
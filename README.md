# toponets

A simple interface to load checkpoints from the [TopoNets Paper](https://toponets.github.io) (ICLR 2025). We closely followed the standard training procedures for these models. The only difference being the addition of [topoloss](https://github.com/toponets/topoloss) to the original loss.

All of the checkpoints are hosted on the [MurtyLab HuggingFace](https://huggingface.co/murtylab) 🤗 

Start by installing this repo as a module:
```
pip install git+https://github.com/toponets/toponets.git
```

Then load the pre-trained checkpoints like this:

```python
import toponets

topo_resnet18 = toponets.resnet18(tau=10.0, checkpoint_path = f"resnet18_tau_{tau}.pt")
topo_resnet50 = toponets.resnet50(tau=30.0, checkpoint_path = f"resnet50_tau_{tau}.pt")
topo_vit_b_32 = toponets.vit_b_32(tau=10.0, checkpoint_path = f"vit_b_32_tau_{tau}.pt")
topo_nanogpt = toponets.nanogpt(tau=50.0, checkpoint_path = f"nanogpt_tau_{tau}.pt")
```

Check out [example.py](https://github.com/toponets/toponets/blob/main/example.py) for more examples.

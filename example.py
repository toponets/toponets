import toponets

possible_taus = [
    0.5,
    1.0,
    5.0,
    10.0,
    20.0,
    50.0
]
for tau in possible_taus:
    model = toponets.resnet18(tau=tau, checkpoint_path = f"checkpoints/resnet18_tau_{tau}.pt")

possible_taus = [30.0]
for tau in possible_taus:
    model = toponets.resnet50(tau=tau, checkpoint_path = f"checkpoints/resnet50_tau_{tau}.pt")

possible_taus = [10.0]
for tau in possible_taus:
    model = toponets.vit_b_32(tau=tau, checkpoint_path = f"checkpoints/vit_b_32_tau_{tau}.pt")

possible_taus = [
    0.5, 
    1.0,
    3.0,
    50.0
]
for tau in possible_taus:
    model = toponets.nanogpt(tau=tau, checkpoint_path = f"checkpoints/nanogpt_tau_{tau}.pt")
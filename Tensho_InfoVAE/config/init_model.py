import torch

from ..models import InfoVAE
from .constants import get


def setup():
    model = InfoVAE(
        in_channels=1,
        out_channels=1,
        latent_dim=3,
        hidden_dims=get("hidden_dims"),
        # alpha=get("alpha"),
        # beta=get("beta"),
        reg_weight=get("reg_weight"),
        kernel_type=get("kernel_type"),
        latent_var=get("latent_var"),
        scheduler_gamma=get("scheduler_gamma"),
    )

    if get("usecuda"):
        model.cuda(get("idgpu"))
    if get("usemps"):
        model.to(torch.device("mps"))

    return model


def save(name, model):
    torch.save(model.state_dict(), get("model_path")/name)
    print(f"Model saved to {get('model_path')}")

def load(name, model=None):
    if not model:
        model = setup()
    model.load_state_dict(torch.load(get("model_path")/name))
    print(f"Model loaded from {get('model_path')/name}")
    model.eval()
    return model
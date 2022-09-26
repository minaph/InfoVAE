import torch
from torch.utils.data import DataLoader
import numpy as np

from .. import config


def generate_labelled_samples(model: torch.nn.Module, test_loader: DataLoader):
    z_list, label_list = [], []

    for batch_idx, (test_x, test_y) in enumerate(test_loader):
        if batch_idx > 50:
            break
        if config.get("usecuda"):
            test_x = test_x.cuda(config.get("idgpu"))
        if config.get("usemps"):
            test_x = test_x.to(torch.device("mps"))
        mu, logvar = model.encode(test_x)
        z = model.reparameterize(mu, logvar)
        z_list.append(z.data.cpu())
        label_list.append(test_y)
    z = np.concatenate(z_list, axis=0)
    label = np.concatenate(label_list)
    return z, label

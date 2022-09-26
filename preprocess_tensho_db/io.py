from itertools import islice
from pathlib import Path
import numpy as np
from PIL import Image
from torchvision import transforms


def load_img_binaries(dir_path):
    array = []
    for file_path in islice(Path(dir_path).rglob("*.jpg"), 100):
        im = np.array(transforms.Grayscale()(Image.open(file_path)))
        array.append(im.reshape(-1))
    return array
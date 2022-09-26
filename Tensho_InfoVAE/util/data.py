from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data import random_split

IMG_SIDE_LENGTH = 64

def loader(x):
    return transforms.ToTensor()(Image.open(x)).reshape(1, IMG_SIDE_LENGTH, IMG_SIDE_LENGTH)

def make_dataset():
    tensho = datasets.ImageFolder(
        root=f'data/tensho_selected/', 
        loader=loader,
    )
    train_size = int(0.8 * len(tensho))
    valid_size = int(0.1 * len(tensho))
    test_size = len(tensho) - train_size - valid_size

    train_dataset, valid_dataset, test_dataset = random_split(tensho, [train_size, valid_size, test_size])
    del tensho
    return train_dataset, valid_dataset, test_dataset
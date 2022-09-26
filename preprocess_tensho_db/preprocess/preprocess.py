import numpy as np
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import re
from pathlib import Path

from .metadata import allowed_chars
from .binarize import optimized_thresholds

tensho_dir = Path("data/tensho")
tensho_selected_dir = Path("data/tensho_selected")

IMG_SIDE_LENGTH = 64

def preprocess(path: str):
    img = Image.open(path)
    img: Image.Image = transforms.Grayscale()(img)
    # orig_img = img
    np_img = np.array(img)

    te_id = re.search(r"TE\d{5}", path).group(0)
    th = optimized_thresholds.get(te_id)
    if th is None:
        print(f"threshold not found for {te_id}, using 128")
        th = 128

    np_img = np.uint8(np_img < th) * 255
    
    img = Image.fromarray(np_img)
    mass_y, mass_x = np.where(np_img < 128)
    try:
        center_x = int(np.average(mass_x))
        center_y = int(np.average(mass_y))
    except:
        center_x = 0
        center_y = 0
    if not (center_x in range(0, img.size[0]) and center_y in range(0, img.size[1])):
        raise ValueError("center of mass is out of range: ", center_x, center_y, img.size)

    max_len = max(
        center_x,
        center_y,
        img.size[0] - center_x,
        img.size[1] - center_y
    )
    pad_rect = (
        int(max_len - center_x), 
        int(max_len - center_y), 
        int(center_x + max_len - img.size[0]), 
        int(center_y + max_len - img.size[1])
    )
 
    padded = transforms.Pad(pad_rect, fill=0, padding_mode='constant')(img)

    resized = transforms.Resize((IMG_SIDE_LENGTH - 4, IMG_SIDE_LENGTH - 4))(padded)
    result = transforms.Pad(2, fill=0, padding_mode='constant')(resized)
    
    return transforms.ToTensor()(result).reshape(1, IMG_SIDE_LENGTH, IMG_SIDE_LENGTH)


# preprocess
def make_preprocessed_dataset():
    print(f"from: {tensho_dir.absolute()}\nto: {tensho_selected_dir.absolute()}")

    progress = tqdm(tensho_dir.iterdir(), total = len(list(tensho_dir.iterdir())))
    to_pil = transforms.ToPILImage()
    for tensho_db in progress:
        te_id = tensho_db.name
        if te_id not in optimized_thresholds:
            progress.write(f"skipping {te_id}")
            continue
        progress.write(f"processing {te_id}")
        for char_dir in (tensho_db/"characters").iterdir():
            char_id = char_dir.name
            if char_id not in allowed_chars:
                continue
            for img_path in char_dir.iterdir():
                img = preprocess(str(img_path))
                img_dir = tensho_selected_dir/char_id
                
                img_dir.mkdir(parents=True, exist_ok=True)
                to_pil(img).save(img_dir/img_path.name, "JPEG", quality=95)



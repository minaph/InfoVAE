import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from pathlib import Path

from ..io import load_img_binaries
from .metadata import filter_chars

def compute_standard_deviations(imgs: list[np.ndarray]):
    data = []
    for th in trange(256):
        acc_std = 0
        acc_removed = 0
        for img in imgs:
            np_img = np.uint8(img > th) * 255
            acc_std += np.std(np_img)
            acc_removed += np.std(img - np_img)
        data.append((acc_std, acc_removed))
    return data

def compute_standard_deviations_2(imgs: list[np.ndarray]):
    data = []
    for th in trange(256):
        acc_std = 0
        acc_removed = 0
        acc_orig_std = 0
        for img in imgs:
            np_img = np.uint8(img > th) * 255
            acc_std += np.std(np_img)
            acc_removed += np.std(img - np_img)
            acc_orig_std += np.std(img)
        data.append((acc_std, acc_removed, acc_orig_std))
    return data

def best_threshold_maximize_difference_of_std(stds: list[tuple[float, float]]):
    df = pd.DataFrame(stds, columns=["std", "removed"])
    df["removed"] -= df["removed"].min()
    df["removed"] /= df["removed"].max()
    df["std"] /= df["std"].max()
    df["loss"] = df["std"] - df["removed"]
    best_th = df["loss"].idxmin()
    return best_th, df

def optimize_threshold_for_a_book(te_id: str):
    path = f"/Users/life_mac_43/Projects/MediaDesign/MMD-Variational-Autoencoder-Pytorch-InfoVAE/data/tensho/{te_id}/characters/"
    imgs = load_img_binaries(path)
    stds = compute_standard_deviations(imgs)
    best_th, df = best_threshold_maximize_difference_of_std(stds)

    df.plot(y=["std", "removed", "loss"]).plot(title=f"th_{te_id}, best_th={best_th}")
    plt.show()
    return best_th

def optimize_thresholds():
    _, resources = filter_chars(100)
    ths = []
    for te_id in tqdm(resources["te_id"].unique()):
        th = optimize_threshold_for_a_book(te_id)
        ths.append((te_id,th))
    
    return ths

optimized_thresholds_path = Path(__file__).parent / "optimized_thresholds.csv"
if not optimized_thresholds_path.exists():
    ths = optimize_thresholds()
    pd.DataFrame(ths, columns=["te_id", "th"]).to_csv("optimized_thresholds.csv", index=False)
optimized_thresholds = pd.read_csv(optimized_thresholds_path, index_col=0).to_dict().get("th")


def compute_otsu_criteria(im, th):
    # create the thresholded image
    thresholded_im = np.zeros(im.shape)
    thresholded_im[im >= th] = 1

    # compute weights
    nb_pixels = im.size
    nb_pixels1 = np.count_nonzero(th)
    weight1 = nb_pixels1 / nb_pixels
    weight0 = 1 - weight1

    # if one the classes is empty, eg all pixels are below or above the threshold, that threshold will not be considered
    # in the search for the best threshold
    if weight1 == 0 or weight0 == 0:
        return np.nan

    # find all pixels belonging to each class
    val_pixels1 = im[thresholded_im == 1]
    val_pixels0 = im[thresholded_im == 0]

    # compute variance of these classes
    var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0
    var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0

    return weight0 * var0 + weight1 * var1

def optimize_otsu_threshold(im: np.ndarray):
    threshold_range = range(np.max(im)+1)
    criterias = [compute_otsu_criteria(im, th) for th in threshold_range]
    best_threshold = threshold_range[np.argmin(criterias)]

    return best_threshold, criterias
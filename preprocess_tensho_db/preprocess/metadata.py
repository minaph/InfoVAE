import pandas as pd
from tqdm import trange
import os
from pathlib import Path

def assemble_metadata_csv():
    resources = []
    unicodes = pd.DataFrame(columns=["Image"], index=pd.Series(name="Unicode", dtype=object), dtype=int)
    for book_id in trange(1, 40):
        te_id = f"TE{book_id:05d}"
        path = f"/Users/life_mac_43/Projects/MediaDesign/MMD-Variational-Autoencoder-Pytorch-InfoVAE/data/tensho/{te_id}/{te_id}_coordinate.csv"
        if not os.path.exists(path):
            print(f"skipping {te_id}")
            continue
        df = pd.read_csv(path)
        counts = df.groupby("Unicode")["Image"].count()
        for name, value in zip(counts.index, counts):
            if name not in unicodes.index:
                unicodes.loc[name] = value
            else:
                unicodes.loc[name] += value
            resources.append((te_id, name, value))
    resources = pd.DataFrame(resources, columns=["te_id", "char_id", "count"])
    return unicodes, resources

def filter_chars(count_threshold: int):
    unicodes, resources = assemble_metadata_csv()
    unicodes = unicodes[unicodes["Image"] > count_threshold]
    resources = resources[resources["char_id"].isin(unicodes.index)]
    # unicodes.to_csv("allowed_chars.csv")
    return unicodes, resources

chars_list_path = Path(__file__).parent / "allowed_chars.csv"
if not chars_list_path.exists():
    chars, _ = filter_chars(100)
    chars.to_csv(chars_list_path)
allowed_chars = pd.read_csv(chars_list_path, index_col=0).index
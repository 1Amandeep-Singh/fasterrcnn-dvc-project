"""
DVC stage: `prepare`

Reads the raw COCO-format annotations file, splits it into train/val subsets
by image (so all annotations for one image stay together), and writes two
new COCO-format json files into data/processed/. This is the DVC-tracked
output that the `train` and `evaluate` stages depend on.
"""
import argparse
import json
import random
from pathlib import Path

from utils import load_config, set_seed


def split_coco(coco: dict, train_split: float, seed: int):
    images = coco["images"]
    random.Random(seed).shuffle(images)

    n_train = int(len(images) * train_split)
    train_images = images[:n_train]
    val_images = images[n_train:]

    train_ids = {img["id"] for img in train_images}
    val_ids = {img["id"] for img in val_images}

    train_anns = [a for a in coco["annotations"] if a["image_id"] in train_ids]
    val_anns = [a for a in coco["annotations"] if a["image_id"] in val_ids]

    train_coco = {"images": train_images, "annotations": train_anns, "categories": coco["categories"]}
    val_coco = {"images": val_images, "annotations": val_anns, "categories": coco["categories"]}
    return train_coco, val_coco


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["data"]["seed"])

    raw_ann_path = cfg["data"]["raw_annotations"]
    processed_dir = Path(cfg["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    with open(raw_ann_path, "r") as f:
        coco = json.load(f)

    train_coco, val_coco = split_coco(coco, cfg["data"]["train_split"], cfg["data"]["seed"])

    with open(processed_dir / "train.json", "w") as f:
        json.dump(train_coco, f)
    with open(processed_dir / "val.json", "w") as f:
        json.dump(val_coco, f)

    print(f"[prepare] {len(train_coco['images'])} train images, "
          f"{len(val_coco['images'])} val images written to {processed_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    main(args.config)

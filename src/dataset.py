"""COCO-format object detection dataset for Faster R-CNN.

Expects, per split, a json file shaped like:
{
  "images": [{"id": 1, "file_name": "img001.jpg", "width": 640, "height": 480}, ...],
  "annotations": [
    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [x, y, w, h], "iscrowd": 0}, ...
  ],
  "categories": [{"id": 1, "name": "widget"}, ...]
}

This is the standard COCO annotation shape, so any COCO-style labeling tool
(CVAT, Label Studio, Roboflow exports, etc.) can feed this pipeline directly.
"""
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


class CocoDetectionDataset(Dataset):
    def __init__(self, images_dir: str, annotations_file: str, train: bool = True):
        self.images_dir = Path(images_dir)
        with open(annotations_file, "r") as f:
            coco = json.load(f)

        self.images = {img["id"]: img for img in coco["images"]}
        self.image_ids = list(self.images.keys())

        # group annotations by image_id for fast lookup
        self.anns_by_image = {}
        for ann in coco["annotations"]:
            self.anns_by_image.setdefault(ann["image_id"], []).append(ann)

        # category_id -> contiguous label index (0 is reserved for background)
        self.categories = sorted(coco["categories"], key=lambda c: c["id"])
        self.cat_id_to_label = {c["id"]: i + 1 for i, c in enumerate(self.categories)}
        self.label_to_name = {i + 1: c["name"] for i, c in enumerate(self.categories)}
        self.train = train

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.images[image_id]
        img_path = self.images_dir / img_info["file_name"]
        image = Image.open(img_path).convert("RGB")

        anns = self.anns_by_image.get(image_id, [])
        boxes, labels, areas, iscrowd = [], [], [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])          # COCO xywh -> xyxy for torchvision
            labels.append(self.cat_id_to_label[ann["category_id"]])
            areas.append(w * h)
            iscrowd.append(ann.get("iscrowd", 0))

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([image_id]),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }

        image = F.to_tensor(image)  # scales to [0,1], CHW -- matches torchvision's expected input
        return image, target

    @property
    def num_classes(self) -> int:
        return len(self.categories) + 1  # + background

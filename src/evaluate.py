"""
DVC stage: `evaluate`

Runs the trained model over the held-out val split and computes standard
COCO detection metrics (mAP @[.5:.95], mAP@0.5, mAP@0.75, AR) using
pycocotools -- the same metric convention used in the original Faster R-CNN
paper and most published benchmarks. Results are written to
metrics/eval_metrics.json so `dvc metrics show` / `dvc metrics diff` and
CI/CD gates can consume them.
"""
import argparse
import json
from pathlib import Path

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader

from dataset import CocoDetectionDataset
from model import build_model
from utils import collate_fn, get_device, load_config, write_json


@torch.no_grad()
def run_inference(model, loader, device, score_threshold: float):
    model.eval()
    coco_results = []

    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for target, output in zip(targets, outputs):
            image_id = int(target["image_id"].item())
            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                if score < score_threshold:
                    continue
                x1, y1, x2, y2 = box.tolist()
                coco_results.append({
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],   # back to COCO xywh
                    "score": float(score),
                })
    return coco_results


def main(config_path: str):
    cfg = load_config(config_path)
    device = get_device(cfg["train"]["device"])

    processed_dir = Path(cfg["data"]["processed_dir"])
    val_ann_path = processed_dir / "val.json"

    val_ds = CocoDetectionDataset(
        images_dir=cfg["data"]["raw_images_dir"],
        annotations_file=val_ann_path,
        train=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=cfg["train"]["num_workers"], collate_fn=collate_fn,
    )

    model = build_model(cfg["train"]["num_classes"], pretrained_backbone=False).to(device)
    model.load_state_dict(torch.load(cfg["paths"]["model_out"], map_location=device))

    predictions = run_inference(model, val_loader, device, cfg["evaluate"]["score_threshold"])

    if not predictions:
        write_json({"warning": "no predictions above score_threshold", "predictions": 0},
                    cfg["paths"]["eval_metrics"])
        print("[evaluate] no predictions produced -- check score_threshold / model weights")
        return

    coco_gt = COCO(str(val_ann_path))
    coco_dt = coco_gt.loadRes(predictions)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metric_names = [
        "mAP_50_95", "mAP_50", "mAP_75", "mAP_small", "mAP_medium", "mAP_large",
        "AR_max1", "AR_max10", "AR_max100", "AR_small", "AR_medium", "AR_large",
    ]
    results = {name: float(val) for name, val in zip(metric_names, coco_eval.stats)}
    results["num_predictions"] = len(predictions)
    results["num_val_images"] = len(val_ds)

    write_json(results, cfg["paths"]["eval_metrics"])
    print(f"[evaluate] metrics written to {cfg['paths']['eval_metrics']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    main(args.config)

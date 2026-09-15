"""
DVC stage: `train`

Trains torchvision's Faster R-CNN on the processed dataset, logging every
loss component to TensorBoard so training can be watched live with:

    tensorboard --logdir logs/tensorboard

Outputs (all DVC-tracked via dvc.yaml):
  - models/fasterrcnn_model.pth   (final weights + label map)
  - models/label_map.json         (class index -> class name, used by the API)
  - metrics/train_metrics.json    (DVC metrics: final losses per epoch)
  - logs/tensorboard/              (DVC plots: scalar loss curves)
"""
import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import CocoDetectionDataset
from model import build_model
from utils import collate_fn, get_device, load_config, set_seed, write_json


def train_one_epoch(model, optimizer, loader, device, writer, epoch, log_every):
    model.train()
    running_losses = {}
    step = epoch * len(loader)

    for i, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)          # dict of 4 loss components
        total_loss = sum(loss_dict.values())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        for k, v in loss_dict.items():
            running_losses[k] = running_losses.get(k, 0.0) + v.item()

        if i % log_every == 0:
            writer.add_scalar("train/total_loss", total_loss.item(), step + i)
            for k, v in loss_dict.items():
                writer.add_scalar(f"train/{k}", v.item(), step + i)
            print(f"  epoch {epoch} step {i}/{len(loader)}  total_loss={total_loss.item():.4f}")

    return {k: v / len(loader) for k, v in running_losses.items()}


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["data"]["seed"])

    device = get_device(cfg["train"]["device"])
    print(f"[train] using device: {device}")

    processed_dir = Path(cfg["data"]["processed_dir"])
    train_ds = CocoDetectionDataset(
        images_dir=cfg["data"]["raw_images_dir"],
        annotations_file=processed_dir / "train.json",
        train=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate_fn,
    )

    num_classes = cfg["train"]["num_classes"]
    model = build_model(num_classes, cfg["train"]["pretrained_backbone"]).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=cfg["train"]["learning_rate"],
        momentum=cfg["train"]["momentum"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg["train"]["lr_step_size"], gamma=cfg["train"]["lr_gamma"]
    )

    writer = SummaryWriter(log_dir=cfg["paths"]["tensorboard_log_dir"])

    epoch_metrics = []
    start = time.time()
    for epoch in range(cfg["train"]["epochs"]):
        avg_losses = train_one_epoch(
            model, optimizer, train_loader, device, writer, epoch, cfg["train"]["log_every_n_steps"]
        )
        lr_scheduler.step()
        writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], epoch)
        epoch_metrics.append({"epoch": epoch, **avg_losses})
        print(f"[train] epoch {epoch} done, avg total_loss="
              f"{sum(avg_losses.values()):.4f}")

        if (epoch + 1) % cfg["train"]["checkpoint_every"] == 0:
            Path(cfg["paths"]["model_out"]).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), cfg["paths"]["model_out"])

    writer.close()
    elapsed = time.time() - start

    # Final model save + label map for the FastAPI service
    torch.save(model.state_dict(), cfg["paths"]["model_out"])
    write_json(train_ds.label_to_name, cfg["paths"]["label_map"])

    # DVC metrics file -- `dvc metrics show` / `dvc metrics diff` read this
    write_json(
        {
            "epochs": cfg["train"]["epochs"],
            "final_total_loss": sum(epoch_metrics[-1].values()) if epoch_metrics else None,
            "per_epoch": epoch_metrics,
            "train_time_seconds": round(elapsed, 1),
        },
        cfg["paths"]["train_metrics"],
    )
    print(f"[train] finished in {elapsed:.1f}s, model saved to {cfg['paths']['model_out']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    main(args.config)

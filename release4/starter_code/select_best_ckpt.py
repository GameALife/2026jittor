#!/usr/bin/env python3
"""Evaluate existing epoch checkpoints on the validation set and pick the best.

Usage (from starter_code/):
  python select_best_ckpt.py
  python select_best_ckpt.py --every 5          # only epochs 0,5,10,...
  python select_best_ckpt.py --epochs 20 30 45  # explicit list
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import jittor as jt
import numpy as np

jt.flags.use_cuda = 1

from omegaconf import OmegaConf
from tqdm import tqdm

from src.data.dataset import DatasetConfig, PCDatasetModule
from src.data.transform import Transform
from src.model.parse import get_model
from src.system.parse import get_system


def load_yaml(path: str):
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def mean_val_loss(system) -> float:
    system.model.eval()
    system.model.set_predict(False)
    system.on_validation_epoch_start()
    validate_dataloader = system.dataset_module.validate_dataloader()
    assert validate_dataloader is not None
    if not isinstance(validate_dataloader, dict):
        validate_dataloader = {"val": validate_dataloader}
    for name, dataloader in validate_dataloader.items():
        for batch in tqdm(dataloader, desc=f"val/{name}", leave=False):
            system.validation_step(batch)
    sums = []
    for key, vals in system._validation_loss.items():
        if key.endswith("_loss_sum") and vals:
            sums.extend(vals)
    if not sums:
        return float("inf")
    return float(sum(sums) / len(sums))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", default="experiments/vm")
    parser.add_argument("--ckpt_prefix", default="checkpoint")
    parser.add_argument("--every", type=int, default=1, help="evaluate every N epochs")
    parser.add_argument("--epochs", type=int, nargs="*", default=None)
    parser.add_argument("--model_cfg", default="configs/model/vm_legacy_infer.yaml")
    parser.add_argument("--data_cfg", default="configs/data/train.yaml")
    parser.add_argument("--transform_cfg", default="configs/transform/vm.yaml")
    parser.add_argument("--system_cfg", default="configs/system/vm.yaml")
    args = parser.parse_args()

    def _epoch_of(path: str):
        m = re.search(r"_(\d+)\.pkl$", path)
        return int(m.group(1)) if m else None

    ckpt_paths = [
        p for p in glob.glob(os.path.join(args.ckpt_dir, f"{args.ckpt_prefix}_*.pkl"))
        if _epoch_of(p) is not None
    ]
    ckpt_paths.sort(key=_epoch_of)
    if args.epochs is not None:
        want = set(args.epochs)
        ckpt_paths = [p for p in ckpt_paths if _epoch_of(p) in want]
    else:
        ckpt_paths = [p for p in ckpt_paths if _epoch_of(p) % args.every == 0]
    assert ckpt_paths, "no checkpoints found"

    data_config = load_yaml(args.data_cfg)
    transform_config = load_yaml(args.transform_cfg)
    model_config = load_yaml(args.model_cfg)
    system_config = load_yaml(args.system_cfg)

    validate_dataset_config = DatasetConfig.parse(**data_config["validate_dataset"]).split_by_cls()
    model = get_model(model_config=model_config, transform_config=transform_config)
    validate_transform = model.get_validate_transform()
    dataset_module = PCDatasetModule(
        process_fn=model._process_fn,
        train_dataset_config=None,
        validate_dataset_config=validate_dataset_config,
        predict_dataset_config=None,
        train_transform=None,
        validate_transform=validate_transform,
        predict_transform=None,
        debug=False,
    )
    system = get_system(
        dataset_module=dataset_module,
        model=model,
        loss_config={"loss": 1.0},
        optimizer_config=None,
        trainer_config={"epochs": 1},
        writer=None,
        **system_config,
    )

    results = []
    best_loss, best_path = float("inf"), None
    for path in ckpt_paths:
        epoch = int(re.search(r"_(\d+)\.pkl$", path).group(1))
        print(f"\n=== evaluating {path} ===")
        model.load(path)
        loss = mean_val_loss(system)
        results.append((epoch, loss, path))
        print(f"epoch={epoch} val_loss={loss:.6f}")
        if loss < best_loss:
            best_loss, best_path = loss, path

    assert best_path is not None
    best_out = os.path.join(args.ckpt_dir, f"{args.ckpt_prefix}_best.pkl")
    # copy weights file
    import shutil
    shutil.copy2(best_path, best_out)
    meta = os.path.join(args.ckpt_dir, f"{args.ckpt_prefix}_best_meta.txt")
    with open(meta, "w") as f:
        ep = int(re.search(r"_(\d+)\.pkl$", best_path).group(1))
        f.write(f"source={best_path}\n")
        f.write(f"epoch={ep}\n")
        f.write(f"val_loss={best_loss:.8f}\n")
        f.write("ranking (epoch, val_loss):\n")
        for epoch, loss, path in sorted(results, key=lambda x: x[1]):
            f.write(f"  {epoch}\t{loss:.8f}\t{path}\n")
    print(f"\nBEST: {best_path} (val_loss={best_loss:.6f})")
    print(f"copied → {best_out}")
    print(f"meta → {meta}")


if __name__ == "__main__":
    main()

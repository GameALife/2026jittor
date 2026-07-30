#!/usr/bin/env python3
"""Pick nearby epoch checkpoints around best for predict-time ensemble.

Usage (from starter_code/):
  python pick_ensemble.py
  python pick_ensemble.py --radius 2 --ckpt-dir experiments/vm_v2
  python pick_ensemble.py --epochs 23 25 27

Prints comma-separated paths to stdout (for ENSEMBLE_CKPTS=...).
"""
from __future__ import annotations

import argparse
import os
import re


def parse_best_epoch(meta_path: str) -> int | None:
    if not os.path.isfile(meta_path):
        return None
    with open(meta_path) as f:
        for line in f:
            m = re.match(r'epoch=(\d+)', line.strip())
            if m:
                return int(m.group(1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', default='experiments/vm_v2')
    ap.add_argument('--radius', type=int, default=2,
                    help='Include best±radius epoch ckpts (plus best.pkl)')
    ap.add_argument('--epochs', type=int, nargs='*', default=None,
                    help='Explicit epoch list (overrides --radius)')
    ap.add_argument('--include-best', action='store_true', default=True)
    ap.add_argument('--no-include-best', action='store_false', dest='include_best')
    args = ap.parse_args()

    ckpt_dir = args.ckpt_dir
    paths: list[str] = []

    best_pkl = os.path.join(ckpt_dir, 'checkpoint_best.pkl')
    best_ep = parse_best_epoch(os.path.join(ckpt_dir, 'checkpoint_best_meta.txt'))

    if args.include_best and os.path.isfile(best_pkl):
        paths.append(best_pkl)

    if args.epochs is not None:
        epochs = sorted(set(args.epochs))
    else:
        if best_ep is None:
            existing = []
            for name in os.listdir(ckpt_dir):
                m = re.match(r'checkpoint_(\d+)\.pkl$', name)
                if m:
                    existing.append(int(m.group(1)))
            if not existing and not paths:
                print('')
                return
            best_ep = max(existing) if existing else None
        if best_ep is None:
            print(','.join(paths))
            return
        r = max(0, args.radius)
        epochs = list(range(best_ep - r, best_ep + r + 1))

    for ep in epochs:
        # skip duplicate of best.pkl when meta epoch matches
        if args.include_best and best_ep is not None and ep == best_ep:
            continue
        p = os.path.join(ckpt_dir, f'checkpoint_{ep}.pkl')
        if os.path.isfile(p) and p not in paths:
            paths.append(p)

    print(','.join(paths))


if __name__ == '__main__':
    main()

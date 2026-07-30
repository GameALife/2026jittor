"""Load CRAFT checkpoint(s) and generate competition predictions (supports ensemble)."""
import os
import os.path as osp
import sys
import json

root = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, root)
sys.path.insert(0, osp.join(root, 'JittorGeometric'))

os.environ.setdefault('JT_SYNC', '0')

import argparse
import jittor as jt
import numpy as np
import pandas as pd
from jittor_geometric.data import TemporalData
from jittor_geometric.dataloader.temporal_dataloader import get_neighbor_sampler

from config import get_dataset_config
from main import (
    build_craft_model, test_competition,
    build_dst_log_popularity, build_src_dst_cooccur,
)

jt.flags.use_cuda = 1

_INFER_KEYS = (
    'history_boost', 'history_count_coef', 'history_recency_coef', 'cooccur_boost',
    'pop_prior_alpha', 'pred_normalize', 'pred_temperature',
    'ensemble_best_weight', 'ensemble_last_weight',
)


def load_run_config(dataset, save_dir, explicit_path=None, refresh_infer=True, use_tuned=True):
    defaults = get_dataset_config(dataset)
    if explicit_path and osp.exists(explicit_path):
        with open(explicit_path) as f:
            cfg = json.load(f)
    else:
        auto_path = osp.join(save_dir, f'{dataset}_config.json')
        if osp.exists(auto_path):
            with open(auto_path) as f:
                print(f'Loaded run config from {auto_path}')
                cfg = json.load(f)
        else:
            print('No saved config found, using dataset defaults from config.py')
            cfg = dict(defaults)
    if refresh_infer:
        for k in _INFER_KEYS:
            if k in defaults:
                cfg[k] = defaults[k]
    # Prefer Val@100 tuned knobs if present
    tuned_path = osp.join(save_dir, f'{dataset}_infer_best.json')
    if use_tuned and osp.exists(tuned_path):
        with open(tuned_path) as f:
            tuned = json.load(f)
        infer = tuned.get('infer', tuned)
        for k, v in infer.items():
            cfg[k] = v
        print(f'Loaded tuned infer from {tuned_path} (val MRR@100={tuned.get("mrr_at_100")})')
    if refresh_infer or (use_tuned and osp.exists(tuned_path)):
        print(
            'Inference knobs: '
            f'history_boost={cfg.get("history_boost")}, '
            f'count={cfg.get("history_count_coef")}, '
            f'recency={cfg.get("history_recency_coef")}, '
            f'cooccur={cfg.get("cooccur_boost")}, '
            f'pop={cfg.get("pop_prior_alpha")}, '
            f'temp={cfg.get("pred_temperature")}'
        )
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Single checkpoint (default: ensemble best+last)')
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--no_refresh_infer', action='store_true')
    parser.add_argument('--no_tuned', action='store_true', help='Ignore saved_models/*_infer_best.json')
    parser.add_argument('--no_ensemble', action='store_true',
                        help='Use only best (or --model_path) without last ckpt')
    parser.add_argument('--pack', action='store_true', help='Also refresh result.zip if both csvs exist')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.data_dir

    cfg = load_run_config(
        args.dataset, args.save_dir, args.config_path,
        refresh_infer=not args.no_refresh_infer,
        use_tuned=not args.no_tuned)
    if args.batch_size is not None:
        cfg['batch_size'] = args.batch_size

    df = pd.read_csv(f'{args.data_dir}/{args.dataset}/train.csv')
    test_df = pd.read_csv(f'{args.data_dir}/{args.dataset}/test.csv')

    src_np = df['src'].values.astype(np.int32)
    dst_np = df['dst'].values.astype(np.int32)
    t_np = df['time'].values.astype(np.int32)
    edge_ids_np = np.arange(len(df), dtype=np.int32) + 1

    test_src = test_df['src'].values.astype(np.int32)
    test_time = test_df['time'].values.astype(np.int32)
    test_candidates = test_df.iloc[:, 2:].values.astype(np.int32)

    full_data = TemporalData(
        src=jt.Var(src_np), dst=jt.Var(dst_np), t=jt.Var(t_np),
        edge_ids=jt.Var(edge_ids_np))
    full_neighbor_sampler = get_neighbor_sampler(full_data, 'recent', seed=1)

    max_node = max(int(src_np.max()), int(dst_np.max()), int(test_candidates.max()))
    node_size = max_node + 1
    dst_min = min(int(dst_np.min()), int(test_candidates.min()))
    src_min = int(src_np.min())

    best_path = osp.join(args.save_dir, f'{args.dataset}_CRAFT_best.pkl')
    last_path = osp.join(args.save_dir, f'{args.dataset}_CRAFT.pkl')

    models = []
    weights = []
    if args.model_path:
        paths = [args.model_path]
        weights = [1.0]
    elif args.no_ensemble:
        paths = [best_path if osp.exists(best_path) else last_path]
        weights = [1.0]
    else:
        paths = []
        weights = []
        if osp.exists(best_path):
            paths.append(best_path)
            weights.append(float(cfg.get('ensemble_best_weight', 0.65)))
        if osp.exists(last_path) and last_path != best_path:
            # Always include last if it is a distinct path (even if same size)
            if last_path not in paths:
                paths.append(last_path)
                weights.append(float(cfg.get('ensemble_last_weight', 0.35)))
        if not paths:
            raise FileNotFoundError(f'No checkpoints under {args.save_dir} for {args.dataset}')
        # If best==last content-wise identical weights still ok; if only one path, weight=1
        if len(paths) == 1:
            weights = [1.0]

    for p in paths:
        m = build_craft_model(cfg, node_size)
        m.set_min_idx(src_min, dst_min)
        m.load_state_dict(jt.load(p))
        models.append(m)
        print(f'Loaded model from {p}')
    print(f'Ensemble weights: {weights}')
    print(
        f'Infer: history={cfg.get("history_boost")}, count={cfg.get("history_count_coef")}, '
        f'recency={cfg.get("history_recency_coef")}, cooccur={cfg.get("cooccur_boost")}, '
        f'pop={cfg.get("pop_prior_alpha")}, temp={cfg.get("pred_temperature")}'
    )

    pop_prior = None
    if float(cfg.get('pop_prior_alpha', 0.0) or 0.0) > 0:
        pop_prior = build_dst_log_popularity(dst_np, node_size)
    cooccur = None
    if float(cfg.get('cooccur_boost', 0.0) or 0.0) > 0:
        print('Building src-dst co-occurrence from train...')
        cooccur = build_src_dst_cooccur(src_np, dst_np)

    scores = test_competition(
        models[0], test_src, test_time, test_candidates,
        full_neighbor_sampler, cfg['num_neighbors'], cfg['batch_size'],
        cfg=cfg, pop_prior=pop_prior, cooccur=cooccur,
        models=models, model_weights=weights,
    )
    print(f'Scores shape: {scores.shape}, range: [{scores.min():.6f}, {scores.max():.6f}]')

    output_file = f'{args.output_dir}/{args.dataset}/{args.dataset}_result.csv'
    os.makedirs(osp.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for row in scores:
            f.write(','.join([f'{p:.8f}' for p in row]) + '\n')
    print(f'Saved predictions to {output_file}')

    if args.pack:
        d1 = osp.join(args.output_dir, 'dataset1', 'dataset1_result.csv')
        d2 = osp.join(args.output_dir, 'dataset2', 'dataset2_result.csv')
        if osp.exists(d1) and osp.exists(d2):
            import shutil
            shutil.copyfile(d1, './dataset1.csv')
            shutil.copyfile(d2, './dataset2.csv')
            os.system('zip -q -j result.zip dataset1.csv dataset2.csv')
            print('Packed ./result.zip')


if __name__ == '__main__':
    main()

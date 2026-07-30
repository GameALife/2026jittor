"""Per-dataset hyperparameters for CRAFT competition training/inference."""

DEFAULT_CONFIG = {
    'num_neighbors': 30,
    'hidden_size': 64,
    'n_layers': 2,
    'n_heads': 2,
    'hidden_dropout_prob': 0.1,
    'attn_dropout_prob': 0.1,
    'emb_dropout_prob': 0.1,
    'lr': 1e-4,
    'batch_size': 200,
    'num_neg_train': 5,
    'num_neg_val': 99,
    'val_every': 2,
    'input_cat_time_intervals': False,
    'output_cat_time_intervals': True,
    'output_cat_repeat_times': True,
    'use_pos': True,
    'skip_connection': True,
    'loss_type': 'BPR',
    'train_loss': 'bpr',
    'train_softmax_temp': 1.0,
    'hard_neg_ratio': 0.0,
    'pop_hard_ratio': 0.0,
    'pop_top_k': 5000,
    'history_boost': 0.0,
    'history_count_coef': 0.0,
    'history_recency_coef': 0.0,
    'cooccur_boost': 0.0,
    'pop_prior_alpha': 0.0,
    'pred_normalize': 'softmax',
    'pred_temperature': 1.0,
    'sync_every': 50,
    'ensemble_best_weight': 0.65,
    'ensemble_last_weight': 0.35,
}

DATASET_CONFIGS = {
    # High-repeat (~72% history hit): NEVER use history as hard negatives
    'dataset1': {
        **DEFAULT_CONFIG,
        'num_neighbors': 50,
        'hidden_size': 64,
        'num_neg_train': 10,
        'num_neg_val': 99,
        'val_every': 2,
        'input_cat_time_intervals': True,
        'output_cat_repeat_times': True,
        'batch_size': 512,
        'lr': 1e-4,
        'train_loss': 'bpr',
        'train_softmax_temp': 1.0,
        'hard_neg_ratio': 0.0,
        'pop_hard_ratio': 0.0,
        'history_boost': 2.0,
        'history_count_coef': 1.5,
        'history_recency_coef': 1.5,
        'cooccur_boost': 2.0,
        'pop_prior_alpha': 0.8,
        'pred_normalize': 'softmax',
        'pred_temperature': 1.0,
        'sync_every': 80,
        'ensemble_best_weight': 0.7,
        'ensemble_last_weight': 0.3,
    },
    # Bipartite — revert to proven recipe (hidden128 / 30neg / hist-hard only), speed up
    'dataset2': {
        **DEFAULT_CONFIG,
        'num_neighbors': 50,
        'hidden_size': 128,
        'n_layers': 2,
        'n_heads': 2,
        'num_neg_train': 30,
        'num_neg_val': 99,
        'val_every': 2,
        'input_cat_time_intervals': True,
        'output_cat_repeat_times': True,
        'batch_size': 192,          # larger batch for GPU util
        'lr': 1e-4,
        'train_loss': 'softmax',
        'train_softmax_temp': 1.0,
        'hard_neg_ratio': 0.5,      # history hard-neg only (worked before)
        'pop_hard_ratio': 0.0,      # pop-hard hurt Val@100 ranking
        'history_boost': 0.0,
        'history_count_coef': 1.0,
        'history_recency_coef': 0.0,
        'cooccur_boost': 1.2,
        'pop_prior_alpha': 0.0,
        'pred_normalize': 'softmax',
        'pred_temperature': 0.7,
        'sync_every': 60,
        'ensemble_best_weight': 0.6,
        'ensemble_last_weight': 0.4,
    },
}


def get_dataset_config(dataset_name):
    """Return a shallow copy of the config for ``dataset_name``."""
    cfg = DATASET_CONFIGS.get(dataset_name, DEFAULT_CONFIG)
    return dict(cfg)

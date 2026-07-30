from collections import defaultdict
from jittor import optim
from typing import Dict, List, Optional
from tqdm import tqdm

import jittor as jt
import os

from ..data.asset import Asset
from ..data.dataset import PCDatasetModule
from ..model.spec import ModelSpec

def _get_item(x):
    if isinstance(x, jt.Var):
        return x.item()
    return x

def get_optimizer(optimizer_config, model):
    __target__ = optimizer_config.pop('__target__')
    MAPPING = {
        'sgd': optim.SGD,
        'adam': optim.Adam,
    }
    if __target__ not in MAPPING:
        raise ValueError(f"unsupported optimizer: {__target__}")
    OptimizerClass = MAPPING[__target__]
    optimizer = OptimizerClass(model.parameters(), **optimizer_config)
    return optimizer

class DummyWriter():
    
    def __init__(self):
        pass
    
    def write(self, batch, prediction: List[Dict], dataset_module: Optional[PCDatasetModule]=None):
        pass

class DummySystem():
    
    def __init__(
        self,
        dataset_module: PCDatasetModule,
        model: ModelSpec,
        loss_config=None,
        optimizer_config=None,
        trainer_config=None,
        writer: Optional[DummyWriter]=None,
        
        ckpt_save_dir: str="experiments",
        ckpt_save_name: str="checkpoint",
    ):
        self.dataset_module = dataset_module
        self.model = model
        self.loss_config = loss_config
        self.ckpt_save_dir = ckpt_save_dir
        self.ckpt_save_name = ckpt_save_name
        self.writer = writer
        if trainer_config is None:
            trainer_config = {}
        self.epochs = trainer_config.get('epochs', 1)
        # Early-stop / best-ckpt monitor: loss_sum | cd | p2s | loss
        self.monitor = str(trainer_config.get('monitor', 'cd'))
        self.early_stop = int(trainer_config.get('early_stop', 10))
        
        if optimizer_config is not None and model is not None:
            self.optimizer = get_optimizer(optimizer_config, model)
        else:
            self.optimizer = None
        
        self._validation_loss = defaultdict(list)
        self.best_val_metric = float('inf')
        self.best_epoch = -1
        self._current_epoch = -1
        self._epochs_no_improve = 0
        # backward-compat alias
        self.best_val_loss = self.best_val_metric
    
    def forward(self, batch, validate: bool=False): # return loss sum
        loss_dict = self.model.training_step(batch)
        assert isinstance(loss_dict, dict), "loss_dict must be a dict containing loss/metrics"
        assert self.loss_config is not None, "do not have loss_confing"
        loss_sum = 0.
        if validate:
            assets: List[Asset] = [a for a in batch['asset']]
            cls = assets[0].cls # guaranteed to be the same cls in dataloader
            for name, val in loss_dict.items():
                # Always record; only weight keys present in loss_config
                self._validation_loss[f"val/{cls}_{name}"].append(_get_item(val))
                if name in self.loss_config:
                    loss_sum = loss_sum + self.loss_config[name] * val
            self._validation_loss[f"val/{cls}_loss_sum"].append(_get_item(loss_sum))
        else:
            for name, val in loss_dict.items():
                if name not in self.loss_config:
                    continue
                if self.loss_config[name] > 0:
                    loss_sum = loss_sum + self.loss_config[name] * val
            loss_dict['loss_sum'] = loss_sum
        if not isinstance(loss_sum, jt.Var):
            return jt.array(loss_sum)
        return loss_sum
    
    def on_train_epoch_start(self):
        pass
    
    def on_train_batch_start(self):
        pass
    
    def training_step(self, batch):
        return self.forward(batch, validate=False)
    
    def on_train_batch_end(self):
        pass
    
    def on_train_epoch_end(self):
        pass
    
    def on_validation_epoch_start(self):
        self._validation_loss = defaultdict(list)
    
    def on_validation_batch_start(self):
        pass
    
    def validation_step(self, batch):
        assert self.loss_config is not None, "do not have loss_confing"
        return self.forward(batch, validate=True)
    
    def on_validation_batch_end(self):
        pass

    def _aggregate_monitor(self) -> Optional[float]:
        """Mean of monitored metric over all val batches / classes. Lower is better."""
        mon = self.monitor
        if mon == 'loss_sum':
            suffix = '_loss_sum'
        else:
            suffix = f'_{mon}'
        vals = []
        for key, arr in self._validation_loss.items():
            if key.endswith(suffix) and arr:
                vals.extend(arr)
        if not vals:
            return None
        return float(sum(vals) / len(vals))
    
    def on_validation_epoch_end(self):
        mean_val = self._aggregate_monitor()
        if mean_val is None:
            # fallback to loss_sum
            sums = []
            for key, vals in self._validation_loss.items():
                if key.endswith('_loss_sum') and vals:
                    sums.extend(vals)
            if not sums:
                print('[val] no metrics recorded, skip best-ckpt update')
                return False
            mean_val = float(sum(sums) / len(sums))
            mon_name = 'loss_sum'
        else:
            mon_name = self.monitor

        # also log cd/p2s/loss if present
        extras = []
        for name in ('cd', 'p2s', 'repulse', 'loss', 'loss_sum'):
            vs = []
            suf = f'_{name}'
            for key, arr in self._validation_loss.items():
                if key.endswith(suf) and arr:
                    vs.extend(arr)
            if vs:
                extras.append(f'{name}={sum(vs)/len(vs):.6f}')
        extra_str = (' | ' + ' '.join(extras)) if extras else ''

        print(f'[val] epoch={self._current_epoch} monitor={mon_name}={mean_val:.6f} '
              f'(best={self.best_val_metric:.6f} @ epoch {self.best_epoch}){extra_str}')

        improved = mean_val < self.best_val_metric
        if improved:
            self.best_val_metric = mean_val
            self.best_val_loss = mean_val
            self.best_epoch = self._current_epoch
            self._epochs_no_improve = 0
            os.makedirs(self.ckpt_save_dir, exist_ok=True)
            best_path = os.path.join(self.ckpt_save_dir, f'{self.ckpt_save_name}_best.pkl')
            self.model.save(best_path)
            meta_path = os.path.join(self.ckpt_save_dir, f'{self.ckpt_save_name}_best_meta.txt')
            with open(meta_path, 'w') as f:
                f.write(f'epoch={self.best_epoch}\n')
                f.write(f'monitor={mon_name}\n')
                f.write(f'val_metric={self.best_val_metric:.8f}\n')
                for e in extras:
                    f.write(f'{e}\n')
            print(f'[val] new best → saved {best_path}')
        else:
            self._epochs_no_improve += 1
            print(f'[val] no improve for {self._epochs_no_improve} epoch(s) '
                  f'(patience={self.early_stop})')
        return improved
    
    def on_before_optimizer_step(self, optimizer):
        pass
    
    def on_predict_epoch_start(self):
        pass
    
    def on_predict_batch_start(self):
        pass
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        return self.model.predict_step(batch)
    
    def on_predict_batch_end(self):
        pass
    
    def on_predict_epoch_end(self):
        pass
    
    def train(self):
        assert self.optimizer is not None, "optimizer is None, cannot train"
        self.model.set_predict(False)
        for epoch in range(self.epochs):
            self._current_epoch = epoch
            self.model.train()
            self.on_train_epoch_start()
            train_dataloader = self.dataset_module.train_dataloader()
            assert train_dataloader is not None, "train_dataloader is None"
            pbar = tqdm(train_dataloader, total=len(train_dataloader)//train_dataloader.batch_size) # type: ignore
            for batch in pbar:
                self.on_train_batch_start()
                loss = self.training_step(batch)
                self.optimizer.zero_grad()
                self.optimizer.backward(loss)
                pbar.set_description(f"Epoch {epoch}, Loss: {_get_item(loss)}")
                self.on_before_optimizer_step(self.optimizer)
                self.optimizer.step()
                self.on_train_batch_end()
            self.on_train_epoch_end()
            
            self.model.eval()
            validate_dataloader = self.dataset_module.validate_dataloader()
            if validate_dataloader is not None:
                self.on_validation_epoch_start()
                if isinstance(validate_dataloader, dict):
                    for name, dataloader in validate_dataloader.items():
                        pbar = tqdm(dataloader, total=len(dataloader)//dataloader.batch_size)
                        for batch in pbar:
                            self.on_validation_batch_start()
                            loss = self.validation_step(batch)
                            pbar.set_description(f"Epoch {epoch}, Validate {name}, Loss: {_get_item(loss)}")
                            self.on_validation_batch_end()
                else:
                    pbar = tqdm(validate_dataloader, total=len(validate_dataloader)//validate_dataloader.batch_size)
                    for batch in pbar:
                        self.on_validation_batch_start()
                        loss = self.validation_step(batch)
                        pbar.set_description(f"Epoch {epoch}, Validate, Loss: {_get_item(loss)}")
                        self.on_validation_batch_end()
                self.on_validation_epoch_end()
            
            checkpoint_path = os.path.join(self.ckpt_save_dir, f'{self.ckpt_save_name}_{epoch}.pkl')
            os.makedirs(self.ckpt_save_dir, exist_ok=True)
            self.model.save(checkpoint_path)
            if self.best_epoch >= 0:
                print(f'[train] epoch {epoch} done | best so far: epoch {self.best_epoch} '
                      f'{self.monitor}={self.best_val_metric:.6f}')

            if self.early_stop > 0 and self._epochs_no_improve >= self.early_stop:
                print(f'[train] early stop at epoch {epoch} '
                      f'(best epoch {self.best_epoch}, {self.monitor}={self.best_val_metric:.6f})')
                break
    
    def predict(self):
        # only iterate once
        self.model.set_predict(True)
        self.model.eval()
        self.on_predict_epoch_start()
        predict_dataloader = self.dataset_module.predict_dataloader()
        assert predict_dataloader is not None, "predict_dataloader is None"
        if not isinstance(predict_dataloader, dict):
            predict_dataloader = {"predict": predict_dataloader}
        for dataloader_name, dataloader in predict_dataloader.items():
            pbar = tqdm(dataloader, total=len(dataloader)//dataloader.batch_size) # type: ignore
            for batch_idx, batch in enumerate(pbar):
                self.on_predict_batch_start()
                output = self.predict_step(batch, batch_idx)
                if self.writer is not None:
                    self.writer.write(batch, output, dataset_module=self.dataset_module)
                pbar.set_description(f"Predicting {dataloader_name}, Batch {batch_idx}")

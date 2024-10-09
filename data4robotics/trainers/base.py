# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch, wandb
import numpy as np
from abc import ABC, abstractmethod
from torch.nn.parallel import DistributedDataParallel as DDP


TRAIN_LOG_FREQ, EVAL_LOG_FREQ = 100, 1
class RunningMean:
    def __init__(self, max_len=TRAIN_LOG_FREQ):
        self._values = []
        self._ctr, self._max_len = 0, max_len
    
    def append(self, item):
        self._ctr = (self._ctr + 1) % self._max_len
        if len(self._values) < self._max_len:
            self._values.append(item)
        else:
            self._values[self._ctr] = item
    
    @property
    def mean(self):
        if len(self._values) == 0:
            raise ValueError
        return np.mean(self._values)


class BaseTrainer(ABC):
    def __init__(self, model, device_id):
        self.model, self.device_id = model, device_id
        self.set_device(device_id)
        self.optim = self.configure_optimizers()
        self._trackers = dict()
        self._is_train = True; self.set_train()

    @abstractmethod
    def training_step(self, batch_input, global_step):
        pass

    @abstractmethod
    def configure_optimizers(self):
        pass

    def save_checkpoint(self, save_path, global_step):
        model = self.model
        model_weights = model.module.state_dict() if isinstance(model, DDP) \
                        else model.state_dict()
        save_dict = dict(model=model_weights,
                         optim=self.optim.state_dict(),
                         global_step = global_step)
        torch.save(save_dict, save_path)
    
    def _save_callback(self, save_path, save_dict):
        pass

    def load_checkpoint(self, load_path):
        load_dict = torch.load(load_path)
        model = self.model
        model = model.module if isinstance(model, DDP) \
                else model
        model.load_state_dict(load_dict['model'])
        self.optim.load_state_dict(load_dict['optim'])
        return load_dict['global_step']

    def _load_callback(self, load_path, load_dict):
        pass
    
    def wrap_ddp(self):
        self.model = DDP(model, device_ids=[self.device_id])

    def set_train(self):
        self._is_train = True
        self.model = self.model.train()
    
    def set_eval(self):
        self._is_train = False
        self.model = self.model.eval()
        
        # reset running mean for eval trackers
        for k in self._trackers:
            if 'eval/' in k:
                self._trackers[k] = RunningMean()

    def log(self, key, global_step, value):
        log_freq = TRAIN_LOG_FREQ if self._is_train else EVAL_LOG_FREQ
        key_prepend = 'train/' if self._is_train else 'eval/'
        key = key_prepend + key

        if key not in self._trackers:
            self._trackers[key] = RunningMean()

        tracker = self._trackers[key]
        tracker.append(value)

        if global_step % log_freq == 0 and wandb.run is not None:
            wandb.log({key: tracker.mean}, step=global_step)

    def set_device(self, device_id):
        self.model = self.model.to(device_id)

import itertools
import logging
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from transformers import Trainer, TrainerCallback
from sft.trainer import LotteryTicketSFTTrainer
from sft.sft import SFT
from typing import Dict

logger = logging.getLogger(__name__)

# AdapterTrainer

class PatchAdapterTrainer(Trainer):
    
    def __init__(
        self,
        *args,
        evaluate_with_patch=False,
        adapter_path=None,
        cls_weights=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.evaluate_with_patch = evaluate_with_patch
        self.adapter_path = adapter_path
        adapter_config = json.load(open(os.path.join(self.adapter_path,'adapter_config.json')))
        self.adapter_name = adapter_config['name']
        self.cls_weights = None
        if cls_weights is not None:
            self.cls_weights = torch.tensor(cls_weights).float()

    def freeze_adapter(self):
        num_frozen = 0
        for k,v in self.model.named_parameters():
            if self.adapter_name in k:
                num_frozen += 1
                v.requires_grad = False
        print(f'number of params frozen: {num_frozen}')

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if self.cls_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.cls_weights.to(model.device))
        else:
            loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
        
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            logs: Dict[str, float] = {}
            tr_loss_scalar = tr_loss.item()
            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)  
        
        metrics = None
        if self.control.should_evaluate:
            if self.evaluate_with_patch:
                device = self.model.device
                self.model.load_adapter(self.adapter_path)
                self.model.set_active_adapters(self.adapter_name)
                self.model.to(device)
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
                self._report_to_hp_search(trial, epoch, metrics)
                self.model.delete_adapter(self.adapter_name)
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
                self._report_to_hp_search(trial, epoch, metrics)
        
        if self.control.should_save:
            self._save_checkpoint(self.model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
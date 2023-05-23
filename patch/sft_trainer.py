import itertools
import logging
import os
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

class PatchLotteryTicketSFTTrainer(LotteryTicketSFTTrainer):
    
    def __init__(
        self,
        *args,
        evaluate_with_patch=False,
        cls_weights=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.evaluate_with_patch = evaluate_with_patch
        self.diffs = None
        self.sft = None
        self.cls_weights = None
        if cls_weights is not None:
            self.cls_weights = torch.tensor(cls_weights).float()

    def set_diffs(self, diffs):
        self.diffs = diffs
        self.sft = SFT()
        self.sft.diffs = self.diffs

    def set_mask(self, mask):
        _mask = {}
        for k,v in mask.items():
            if k in self.maskable_params:
                _mask[k] = v
        self._mask = _mask
    
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

    def training_step(self, *args, **kwargs):
        loss = super().training_step(*args, **kwargs)

        if self._masking_enabled:
            # set gradients for non-trainable parametres to zero.
            for n, p in self.model.named_parameters():
                if n in self.maskable_params and p.grad is not None and n in self._mask:
                    p.grad *= self._mask[n]
        return loss

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            logs: Dict[str, float] = {}
            tr_loss_scalar = tr_loss.item()
            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            if self._reg_loss != 0.0:
                logs['l1_reg_loss'] = round(self._reg_loss / (self.state.global_step - self._globalstep_last_logged), 4)
                self._reg_loss = 0.0

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)  
        
        metrics = None
        if self.control.should_evaluate:
            if self.evaluate_with_patch:
                model_sd_copy = {k:v.clone() for k,v in self.model.state_dict().items()}
                self.sft.apply(self.model, with_abs=False)
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
                self._report_to_hp_search(trial, epoch, metrics)
                self.model.load_state_dict(model_sd_copy)
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
                self._report_to_hp_search(trial, epoch, metrics)
        
        if self.control.should_save:
            if self.evaluate_with_patch:
                model_sd_copy = {k:v.clone() for k,v in self.model.state_dict().items()}
                self.sft.apply(self.model, with_abs=False)
            
                self._save_checkpoint(self.model, trial, metrics=metrics)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)

                self.model.load_state_dict(model_sd_copy)
            else:
                self._save_checkpoint(self.model, trial, metrics=metrics)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
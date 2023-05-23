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

class PatchTrainer(Trainer):
    
    def __init__(
        self,
        *args,
        cls_weights=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.cls_weights = None
        if cls_weights is not None:
            self.cls_weights = torch.tensor(cls_weights).float()
    
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

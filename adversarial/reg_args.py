from dataclasses import dataclass, field
from typing import Optional

@dataclass
class RegArguments:
    # # root for explanation regularization args
    # self.reg_explanations = bool(getattr(some_args, 'reg_explanations', False))
    # self.reg_strength = some_args.reg_strength if self.reg_explanations else None
    # self.neutral_words_file = some_args.neutral_words_file if self.reg_explanations else None
    # self.expl_config = full_config

    # root for adv debiasing args
    adv_debias: Optional[bool] = field(default=False, metadata={"help": "enable to do adversarial debiasing"})
    finetune: Optional[bool] = field(default=False, metadata={"help": "enable to do finetuning classification"})
    adv_strength: Optional[float] = field(default=1.0, metadata={"help": "weight of adversarial loss"})
    adv_grad_rev_strength: Optional[float] = field(default=1.0, metadata={"help": "weight of reverse grad"})
    adv_attr_dim: Optional[int] = field(default=None, metadata={"help" : "no of attribute labels"})
    adv_layer_num: Optional[int] = field(default=1, metadata={"help" : "no of hidden layers"})
    adv_lr_scale: Optional[float] = field(default=10.0, metadata={"help": "lr scale of adversarial parameters"})
    adv_dropout: Optional[float] = field(default=0.3, metadata={"help": "dropout for adversarial layer"})
    
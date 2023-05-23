import os
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DebiasArguments:
    "Arguments pertaining to sparse fine-tuning configuration."""
    
    debias_configuration: Optional[str] = field(
        default='none', metadata={"help": "which configuration should be used. choose between [none, before, after]"}
    )
    patch_path: str = field(
        default=None, metadata={"help": "Optional path to diffs/adapter"}
    )
    peft: str = field(
        default=None, metadata={"help": "Optional path to diffs/adapter"}
    )
    
    def __post_init__(self):

        assert self.debias_configuration in ['none', 'before', 'after']
        if self.debias_configuration != 'none':
            if self.peft == 'sft':
                self.patch_path = os.path.join(self.patch_path, 'pytorch_diff.bin')
                print(self.patch_path)
                assert os.path.isfile(self.patch_path), self.patch_path
            else:
                assert os.path.isdir(self.patch_path), self.patch_path
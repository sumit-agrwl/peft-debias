# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified by the Cambridge Language Technology Lab
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import random
import json
import copy
from collections import defaultdict

from functools import partial
import datasets
from datasets import load_dataset
import torch
from torch import nn

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    set_seed,
)
from transformers.adapters import AdapterArguments, AdapterTrainer, setup_adapter_training
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from sft import (
    LotteryTicketSFTTrainer,
    SFT,
    SftArguments
)
from adversarial import (
    RegArguments,
    AdvBertForMaskedLM,
    AdvAdapterTrainer,
    AdvLotteryTicketSFTTrainer
)


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.9.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    protected_attribute_column: Optional[str] = field(
        default=None,
        metadata={"help": "Column of the protected attribute"},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    peft: Optional[str] = field(
        default='sft', metadata={"help": "Default peft technique to use."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    patience: Optional[int] = field(
        default=100,
        metadata={
            "help": "Patience for early stopping"
        },
    )
    dropout_debias: bool = field(
        default=False,
        metadata={
            "help": "Whether to apply increased dropout regularization or not. Defaults to `True`."
        },
    )
    counterfactual_augmentation: Optional[str] = field(
        default=None,
        metadata={
            "help": "What type of counterfactual augmentation to apply. Defaults to `None`."
        },
    )
    bias_attribute_words_file: str = field(
        default='data/bias_attribute_words.json',
        metadata={"help": "Directory where all persistent data will be stored."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "jsonl", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "jsonl",  "txt"], "`validation_file` should be a csv, a json or a txt file."


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, SftArguments, TrainingArguments, AdapterArguments, RegArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, sft_args, training_args, adapter_args, reg_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, sft_args, training_args, adapter_args, reg_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name, 
            data_args.dataset_config_name, 
            cache_dir=model_args.cache_dir
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
            )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        if extension == 'jsonl':
            extension = 'json'
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
        logger.info(f"New config: {config}")

    # Apply increased dropout regularized for debiasing if specified.
    # We use the hyperparameters specified in: https://arxiv.org/abs/2010.06032.
    if data_args.dropout_debias:
        logger.info(
            f"Setting dropout hyperparameters for: {model_args.model_name_or_path}."
        )

        if config.model_type in ["bert", "roberta"]:
            config.hidden_dropout_prob = 0.20
            config.attention_probs_dropout_prob = 0.15
        else:
            config.hidden_dropout_prob = 0.05
            config.attention_probs_dropout_prob = 0.05

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    # Preprocessing the datasets.

    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
        if reg_args.adv_debias :
            protected_attribute_list = list(set(raw_datasets["train"][data_args.protected_attribute_column]))
            protected_attribute2id = {l: i for i, l in enumerate(protected_attribute_list)}
    else:
        column_names = raw_datasets["validation"].column_names
        if reg_args.adv_debias :
            protected_attribute_list = list(set(raw_datasets["validation"][data_args.protected_attribute_column]))
            protected_attribute2id = {l: i for i, l in enumerate(protected_attribute_list)}

    if reg_args.adv_debias :
        reg_args.adv_attr_dim = len(protected_attribute2id)

    text_column_name = "text" if "text" in column_names else column_names[0]

    if model_args.model_name_or_path:
        if reg_args.adv_debias:
            model = AdvBertForMaskedLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                reg_args=reg_args
            )
        else:
            model = AutoModelForMaskedLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    else:
        logger.info("Training new model from scratch") 
        if reg_args.adv_debias:
            raise NotImplementedError
        model = AutoModelForMaskedLM.from_config(config)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.

    embedding_size = model.get_input_embeddings().weight.shape[0]


    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.

    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    if data_args.line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if data_args.pad_to_max_length else False

        def tokenize_function(examples):
            # Remove empty lines
            examples[text_column_name] = [
                line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
            ]
            return tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )

        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=[text_column_name],
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset line_by_line",
            )
    else:
        # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
        # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
        # efficient when it receives the `special_tokens_mask`.
        def tokenize_function(examples):
            outputs = tokenizer(
                examples[text_column_name], 
                return_special_tokens_mask=True, 
                truncation=True, 
                padding=True, 
                max_length=max_seq_length
            )
            if reg_args.adv_debias:
                protected_attributes = [protected_attribute2id[i] for i in examples[data_args.protected_attribute_column]]
                outputs['attr'] = protected_attributes
            return outputs
            
        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on every text in dataset",
            )

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= max_seq_length:
                total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
        # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
        # might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

        # with training_args.main_process_first(desc="grouping texts together"):
        #     tokenized_datasets = tokenized_datasets.map(
        #         group_texts,
        #         batched=True,
        #         num_proc=data_args.preprocessing_num_workers,
        #         load_from_cache_file=not data_args.overwrite_cache,
        #         desc=f"Grouping texts in chunks of {max_seq_length}",
        #     )

    def _create_bias_attribute_words(attribute_file, bias_type):
        """Creates list of bias attribute words (e.g., he/she).

        Args:
            attribute_file: Path to the file containing the bias attribute words.
            bias_type: Type of bias attribute words to load. Must be one of
                ["gender", "race", "religion"].

        Notes:
            * We combine each bias attribute word with several punctuation marks.
              The current set of words is *not* exhaustive, however, it should
              cover most occurances.
        """
        with open(attribute_file, "r") as f:
            bias_attribute_words = json.load(f)[bias_type]

        result = bias_attribute_words[:]
        for punctuation in [".", ",", "?", "!", ";", ":"]:
            for words in bias_attribute_words:
                augmented_words = [word + punctuation for word in words]
                result.append(augmented_words)
        return result

    def gender_counterfactual_augmentation(examples, bias_attribute_words):
        """Applies gender counterfactual data augmentation to a batch of examples.

        Notes:
            * We apply CDA after the examples have potentially been grouped.
            * This implementation can be made more efficient by operating on
              token IDs as opposed to text. We currently decode each example
              as it is simpler.
        """
        outputs = []
        for input_ids in examples["input_ids"]:
            # For simplicity, decode each example. It is easier to apply augmentation
            # on text as opposed to token IDs.
            sentence = tokenizer.decode(input_ids)
            words = sentence.split()  # Tokenize based on whitespace.
            augmented_sentence = words[:]

            augmented = False
            for position, word in enumerate(words):
                for male_word, female_word in bias_attribute_words:
                    if male_word == word:
                        augmented = True
                        augmented_sentence[position] = female_word

                    if female_word == word:
                        augmented = True
                        augmented_sentence[position] = male_word

            if augmented:
                augmented_sentence = " ".join(augmented_sentence)
                outputs.append(augmented_sentence)
                outputs.append(sentence)

        # There are potentially no counterfactual examples.
        if not outputs:
            return {"input_ids": [], "attention_mask": []}

        return tokenizer(
            outputs,
            return_special_tokens_mask=True,
            add_special_tokens=False,  # Special tokens are already added.
            truncation=True,
            padding=True,
            max_length=max_seq_length,
        )

    def ternary_counterfactual_augmentation(examples, bias_attribute_words):
        """Applies racial/religious counterfactual data augmentation to a batch of
        examples.

        Notes:
            * We apply CDA after the examples have potentially been grouped.
            * This implementation can be made more efficient by operating on
              token IDs as opposed to text. We currently decode each example
              as it is simpler.
        """
        outputs = []
        for input_ids in examples["input_ids"]:
            # For simplicity, decode each example. It is easier to apply augmentation
            # on text as opposed to token IDs.
            sentence = tokenizer.decode(input_ids)
            words = sentence.split()  # Tokenize based on whitespace.
            augmented_sentence = words[:]

            # Sample the augmentation pairs.
            r1_augmentation_pair = random.choice([1, 2])
            r2_augmentation_pair = random.choice([0, 2])
            r3_augmentation_pair = random.choice([0, 1])

            augmented = False
            for position, word in enumerate(words):
                for augmentation_words in bias_attribute_words:
                    # Implementation here.
                    r1_word, r2_word, r3_word = augmentation_words

                    if r1_word == word:
                        augmented = True
                        augmented_sentence[position] = augmentation_words[
                            r1_augmentation_pair
                        ]

                    if r2_word == word:
                        augmented = True
                        augmented_sentence[position] = augmentation_words[
                            r2_augmentation_pair
                        ]

                    if r3_word == word:
                        augmented = True
                        augmented_sentence[position] = augmentation_words[
                            r3_augmentation_pair
                        ]

            if augmented:
                augmented_sentence = " ".join(augmented_sentence)
                outputs.append(augmented_sentence)
                outputs.append(sentence)

        # There are potentially no counterfactual examples.
        if not outputs:
            return {"input_ids": [], "attention_mask": []}

        return tokenizer(
            outputs,
            return_special_tokens_mask=True,
            add_special_tokens=False,  # Special tokens are already added.
            truncation=True,
            padding=True,
            max_length=max_seq_length,
        )

    def mixed_counterfactual_augmentation(examples, bias_attribute_words):
        """Applies racial/religious counterfactual data augmentation to a batch of
        examples.

        Notes:
            * We apply CDA after the examples have potentially been grouped.
            * This implementation can be made more efficient by operating on
              token IDs as opposed to text. We currently decode each example
              as it is simpler.
        """
        outputs = []
        for input_ids in examples["input_ids"]:
            # For simplicity, decode each example. It is easier to apply augmentation
            # on text as opposed to token IDs.
            sentence = tokenizer.decode(input_ids)
            words = sentence.split()  # Tokenize based on whitespace.
            augmented_sentence = words[:]

            augmented = False
            
            # Sample the augmentation pairs.
            r1_augmentation_pair = random.choice([1, 2])
            r2_augmentation_pair = random.choice([0, 2])
            r3_augmentation_pair = random.choice([0, 1])

            for position, word in enumerate(words):
                for augmentation_words in bias_attribute_words:
                    # Implementation here.
                    
                    if len(augmentation_words)==3:
                        r1_word, r2_word, r3_word = augmentation_words

                        if r1_word == word:
                            augmented = True
                            augmented_sentence[position] = augmentation_words[
                                r1_augmentation_pair
                            ]

                        if r2_word == word:
                            augmented = True
                            augmented_sentence[position] = augmentation_words[
                                r2_augmentation_pair
                            ]

                        if r3_word == word:
                            augmented = True
                            augmented_sentence[position] = augmentation_words[
                                r3_augmentation_pair
                            ]
                    elif augmentation_words==2 :
                        r1_word, r2_word = augmentation_words
                        if r1_word == word:
                            augmented = True
                            augmented_sentence[position] = r2_word

                        if r2_word == word:
                            augmented = True
                            augmented_sentence[position] = r1_word

            if augmented:
                augmented_sentence = " ".join(augmented_sentence)
                outputs.append(augmented_sentence)
                outputs.append(sentence)

        # There are potentially no counterfactual examples.
        if not outputs:
            return {"input_ids": [], "attention_mask": []}

        return tokenizer(
            outputs,
            return_special_tokens_mask=True,
            add_special_tokens=False,  # Special tokens are already added.
            truncation=True,
            padding=True,
            max_length=max_seq_length,
        )
    def all_counterfactual_augmentation(examples, bias_attribute_words):
        """Applies racial/religious counterfactual data augmentation to a batch of
        examples.

        Notes:
            * We apply CDA after the examples have potentially been grouped.
            * This implementation can be made more efficient by operating on
              token IDs as opposed to text. We currently decode each example
              as it is simpler.
        """
        outputs = []
        w2idx = {}
        for idx, words in enumerate(bias_attribute_words):
            for w in words:
                w2idx[w] = idx
        augmentation_words = set(w2idx.keys())
        for input_ids in examples["input_ids"]:
            # For simplicity, decode each example. It is easier to apply augmentation
            # on text as opposed to token IDs.
            sentence = tokenizer.decode(input_ids)
            words = sentence.split()  # Tokenize based on whitespace.
            original_sentence = words[:]
            
            augmented_sentences = [copy.deepcopy(original_sentence)]

            all_matches = defaultdict(list)
            for position, word in enumerate(words):
                if word in augmentation_words:
                    all_matches[word].append(position)

            # gay 0 2
            # muslim 4 6
            # gay, muslim
            # [[0,2], [4,6]]
            # [[words_gay], [words_muslim]

            def get_stats(all_matches):
                    words = []
                    sample_words = []
                    positions = []
                    for word in all_matches:
                        words.append(word)
                        positions.append(all_matches[word])
                        sample = copy.deepcopy(bias_attribute_words[w2idx[word]])
                        sample.remove(word)
                        sample_words.append(sample)
                        return words, positions, sample_words
            
            if len(all_matches) > 0:
            
                words, positions, sample_words = get_stats(all_matches)

                for word, pos, sample in zip(words, positions, sample_words):
                    new_words = sample
                    for w in new_words:
                        new_sentence = copy.deepcopy(original_sentence)
                        for position in pos:
                            new_sentence[position] = w
                        augmented_sentences.append(new_sentence)

                for sent in augmented_sentences:
                    sent = " ".join(sent)
                    outputs.append(sent)
            else:
                outputs.append(" ".join(augmented_sentences[0]))

        # There are potentially no counterfactual examples.
        if not outputs:
            return {"input_ids": [], "attention_mask": []}
        
        return tokenizer(
            outputs,
            return_special_tokens_mask=True,
            add_special_tokens=False,  # Special tokens are already added.
            truncation=True,
            padding=True,
            max_length=max_seq_length,
        )


    if data_args.counterfactual_augmentation is not None:
        if data_args.counterfactual_augmentation not in ["gender", "race", "religion", "group", "sexual_orientation"]:
            raise ValueError("Invalid CDA type: {data_args.counterfactual_augmentation")

        logger.info(f"Applying {data_args.counterfactual_augmentation} CDA.")

        # Load the bias attribute words.
        bias_attribute_words = _create_bias_attribute_words(
            data_args.bias_attribute_words_file,
            bias_type=data_args.counterfactual_augmentation,
        )

        if data_args.counterfactual_augmentation == "gender":
            counterfactual_augmentation_func = partial(
                gender_counterfactual_augmentation,
                bias_attribute_words=bias_attribute_words,
            )
        elif data_args.counterfactual_augmentation in ["group", "race", "religion"]:
            counterfactual_augmentation_func = partial(
                all_counterfactual_augmentation,
                bias_attribute_words=bias_attribute_words,
            )            
        else:
            counterfactual_augmentation_func = partial(
                ternary_counterfactual_augmentation,
                bias_attribute_words=bias_attribute_words,
            )

        tokenized_datasets = tokenized_datasets.map(
            counterfactual_augmentation_func,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Applying counterfactual augmentation",
        )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    
    # Data collator
    # This one will take care of randomly masking the tokens.
    pad_to_multiple_of_8 = data_args.line_by_line and training_args.fp16 and not data_args.pad_to_max_length
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability,
        pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
    )

    if data_args.peft == 'sft':
        embeddings = model.base_model.embeddings.word_embeddings.weight
        if model.base_model_prefix == 'bert':
            lm_head = model.cls
            decoder = model.cls.predictions.decoder
        elif model.base_model_prefix == 'roberta':
            lm_head = model.lm_head
            decoder = model.lm_head.decoder
        else:
            raise ValueError(f'Unsupported model type {model.base_model_prefix}')

        if sft_args.freeze_head:
            decoder.weight = nn.Parameter(
                torch.zeros_like(embeddings).copy_(embeddings)
            )
            for param in lm_head.parameters():
                param.requires_grad = False

        if sft_args.untie_embeddings:
            decoder.weight = nn.Parameter(
                torch.zeros_like(embeddings).copy_(embeddings)
            )

        if sft_args.freeze_decoder:
            decoder.weight = nn.Parameter(
                torch.zeros_like(embeddings).copy_(embeddings)
            )
            decoder.weight.requires_grad = False

        if sft_args.freeze_embeddings:
            embeddings.requires_grad = False

        if sft_args.freeze_layer_norm:
            for n, p in model.named_parameters():
                if 'LayerNorm' in n:
                    p.requires_grad = False

        maskable_params = [
            n for n, p in model.named_parameters()
            if n.startswith(model.base_model_prefix) and p.requires_grad
        ]
        # Initialize our Trainer
        if reg_args.adv_debias : 
            trainer = AdvLotteryTicketSFTTrainer(
                sft_args=sft_args,
                reg_args=reg_args,
                maskable_params=maskable_params,
                model=model,
                args=training_args,
                train_dataset=train_dataset if training_args.do_train else None,
                eval_dataset=eval_dataset if training_args.do_eval else None,
                tokenizer=tokenizer,
                data_collator=data_collator,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=data_args.patience)]
            )
        else:
            trainer = LotteryTicketSFTTrainer(
                sft_args=sft_args,
                maskable_params=maskable_params,
                model=model,
                args=training_args,
                train_dataset=train_dataset if training_args.do_train else None,
                eval_dataset=eval_dataset if training_args.do_eval else None,
                tokenizer=tokenizer,
                data_collator=data_collator,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=data_args.patience)]
            )

    else:
        # Setup adapters
        setup_adapter_training(model, adapter_args, "mlm")
        trainer_class = Trainer
        if adapter_args.train_adapter:
            trainer_class = AdapterTrainer
            if reg_args.adv_debias:
                trainer_class = AdvAdapterTrainer

        if reg_args.adv_debias:
            trainer = trainer_class(
                model=model,
                args=training_args,
                reg_args=reg_args,
                train_dataset=train_dataset if training_args.do_train else None,
                eval_dataset=eval_dataset if training_args.do_eval else None,
                tokenizer=tokenizer,
                data_collator=data_collator,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=data_args.patience)]
            )
        else:
            trainer = trainer_class(
                model=model,
                args=training_args,
                train_dataset=train_dataset if training_args.do_train else None,
                eval_dataset=eval_dataset if training_args.do_eval else None,
                tokenizer=tokenizer,
                data_collator=data_collator,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=data_args.patience)]
            )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        if training_args.local_rank <= 0 and data_args.peft == 'sft':
            trainer.sft().save(training_args.output_dir)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub:
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "fill-mask"}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        trainer.push_to_hub(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

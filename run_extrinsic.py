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

import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import evaluate
import datasets
import numpy as np
from datasets import ClassLabel, load_dataset, load_metric
import torch

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from sft import (
    load_single_dataset,
    SFT,
    SftArguments,
    decode_sparse_tensor
)

from patch import DebiasArguments, PatchLotteryTicketSFTTrainer, PatchAdapterTrainer, PatchTrainer

logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    cls_weights: Optional[str] = field(
        default=None, metadata={"help": "Class weights for unbalanced datasets."}
    )
    multisource_data: Optional[str] = field(
        default=None, metadata={"help": "JSON multi-source dataset descriptor."}
    )
    dataset_name: str = field(
        default=None, metadata={"help": "Name of NLI dtaaset."}
    )
    dataset_config_name: str = field(
        default=None, metadata={"help": "Evaluation language. Also train language if `train_language` is set to None."}
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input train data file (tsv file)."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input validation data file (tsv file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data (tsv file)."},
    )
    text_column: Optional[str] = field(
        default='text',
        metadata={"help": "Column name containing label ."},
    )
    label_column: Optional[str] = field(
        default='label',
        metadata={"help": "Column name containing label ."},
    )

    eval_split: Optional[str] = field(
        default='validation', metadata={"help": "The split to evaluate on."}
    )

    predict_split: Optional[str] = field(
        default='test', metadata={"help": "The split to predict/test on."}
    )

    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
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
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    save_prefix: str =  field(
        default=''
    )

    def __post_init__(self):
        try:
            self.cls_weights = json.loads(self.cls_weights)
        except Exception as e:
            self.cls_weights = None


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    head_path: Optional[str] = field(
        default=None, metadata={"help": "Path to model head."}
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
    do_lower_case: Optional[bool] = field(
        default=False,
        metadata={"help": "arg to indicate if tokenizer should do lower case in AutoTokenizer.from_pretrained()"},
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

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, SftArguments, DebiasArguments, TrainingArguments))
    model_args, data_args, sft_args, debias_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
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
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None

    # Set seed before initializing model.
    set_seed(training_args.seed)

    dataset_descriptor = {}
    if data_args.dataset_name:
        dataset_descriptor['name'] = data_args.dataset_name
        if data_args.dataset_config_name:
            dataset_descriptor['config_name'] = data_args.dataset_config_name
    else:
        if data_args.train_file:
            dataset_descriptor['train_file'] = data_args.train_file
        if data_args.validation_file:
            dataset_descriptor['validation_file'] = data_args.validation_file
        if data_args.test_file:
            dataset_descriptor['test_file'] = data_args.test_file

        if data_args.train_file is not None:
            file_type = data_args.train_file.split('.')[-1]
        elif data_args.validation_file is not None:
            file_type = data_args.validation_file.split('.')[-1]
        else:
            file_type = data_args.test_file.split('.')[-1]
        if file_type == 'jsonl':
            file_type = 'json'

        dataset_descriptor['file_type'] = file_type

        if file_type == 'csv':
            dataset_descriptor['load_kwargs'] = {
                'delimiter': ',',
        }
        

    if not training_args.do_train:
        dataset_descriptor['train_split'] = None
    if not training_args.do_eval:
        dataset_descriptor['validation_split'] = None

    raw_datasets = load_single_dataset(
        dataset_descriptor,
        training_args,
        cache_dir=model_args.cache_dir,
        overwrite_cache=data_args.overwrite_cache,
    )

    def get_labels(dataset):
        label_list = sorted(list(set(dataset[data_args.label_column])))
        return label_list

    if training_args.do_predict:
        predict_dataset = raw_datasets[data_args.predict_split]
        label_list = get_labels(predict_dataset)
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        

    if training_args.do_eval:
        eval_dataset = raw_datasets[data_args.eval_split]
        label_list = get_labels(eval_dataset)
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        

    if training_args.do_train:
        train_dataset = raw_datasets['train']
        label_list = get_labels(train_dataset)
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        
    num_labels = len(label_list)
    label2id = {l: i for i, l in enumerate(label_list)}
    if data_args.cls_weights is not None:
        weights = [1]*len(label2id)
        for label, idx in label2id.items():
            weights[idx] = data_args.cls_weights[str(label)]
        data_args.cls_weights = weights

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task="sst2",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        do_lower_case=model_args.do_lower_case,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if debias_args.debias_configuration != "none":
        if debias_args.peft == 'sft':
            mask = None
            task_ft = SFT()
            tensors = torch.load(debias_args.patch_path, map_location=training_args.device)

            if 'diffs' in tensors:
                diffs = {
                    p: decode_sparse_tensor(d).to_dense()
                    for p, d in tensors['diffs'].items()
                }
            else:
                diffs = {}

            if 'abs' in tensors:
                abs = tensors['abs']
            else:
                abs = {}

            if not diffs and not abs:
                logger.warn(f'Empty SFT {debias_args.patch_path}')

            task_ft.diffs = diffs
            if debias_args.debias_configuration == 'before':
                task_ft.apply(model, with_abs=False)

            mask = {}
            for k,v in diffs.items():
                # unmasking non-debias params
                v_mask = (v==0).to(v.dtype).to(training_args.device)
                mask[k] = v_mask
        elif debias_args.peft is not None and debias_args.debias_configuration == 'before':
            model.load_adapter(debias_args.patch_path)
            adapter_config = json.load(open(os.path.join(debias_args.patch_path,'adapter_config.json')))
            model.set_active_adapters(adapter_config['name'])

    # Preprocessing the datasets
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    def preprocess_function(examples):
        # Tokenize the texts
        tokenized_examples = tokenizer(
            examples[data_args.text_column],
            padding=padding,
            max_length=data_args.max_seq_length,
            truncation=True,
        )
        tokenized_examples['label'] = [
            label2id.get(label, label)
            for label in examples[data_args.label_column]
        ]
        return tokenized_examples

    if training_args.do_train:
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if training_args.do_eval:
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        with training_args.main_process_first(desc="test dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on test dataset",
            )
    
    # Get the metric function
    str_metric = training_args.metric_for_best_model.replace("eval_","")
    if str_metric != "loss":
        metric = evaluate.load(str_metric)

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        if str_metric == "f1":
            return metric.compute(predictions=preds, references=p.label_ids, average="macro")
        return metric.compute(predictions=preds, references=p.label_ids)

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    if sft_args.freeze_layer_norm:
        for n, p in model.named_parameters():
            if 'LayerNorm' in n:
                p.requires_grad = False

    maskable_params = [
        n for n, p in model.named_parameters()
        if n.startswith(model.base_model_prefix) and p.requires_grad
    ]

    # Initialize our Trainer
    if debias_args.debias_configuration == 'none':
        trainer_cls = PatchTrainer
        trainer = trainer_cls(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            cls_weights=data_args.cls_weights,
        )
    else:
        if debias_args.peft == 'sft':
            trainer_cls = PatchLotteryTicketSFTTrainer
            trainer = trainer_cls(
                sft_args=sft_args,
                maskable_params=maskable_params,
                model=model,
                args=training_args,
                train_dataset=train_dataset if training_args.do_train else None,
                eval_dataset=eval_dataset if training_args.do_eval else None,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                evaluate_with_patch=debias_args.debias_configuration == 'after',
                cls_weights=data_args.cls_weights,
            )

            if debias_args.debias_configuration == "after":
                trainer.set_diffs(diffs)

            if mask is not None:
                logger.info('using mask')
                trainer.set_mask(mask)
        else:
            trainer_cls = PatchAdapterTrainer
            trainer = trainer_cls(
                        model=model,
                        args=training_args,
                        train_dataset=train_dataset if training_args.do_train else None,
                        eval_dataset=eval_dataset if training_args.do_eval else None,
                        tokenizer=tokenizer,
                        data_collator=data_collator,
                        compute_metrics=compute_metrics,
                        evaluate_with_patch=debias_args.debias_configuration == 'after',
                        adapter_path=debias_args.patch_path,
                        cls_weights=data_args.cls_weights,
                        )
            if debias_args.debias_configuration == 'before':
                trainer.freeze_adapter()

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        if debias_args.debias_configuration == 'none':
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
        else:
            if debias_args.peft == 'sft':
                train_result = trainer.fine_tune(resume_from_checkpoint=checkpoint)
            else:
                train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        if training_args.local_rank <= 0:
            trainer.save_model(training_args.output_dir)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        if debias_args.peft != 'sft' and debias_args.debias_configuration == 'after':
            model = trainer.model
            model.load_adapter(trainer.adapter_path)
            model.set_active_adapters(trainer.adapter_name)
            model.to(model.device)
            trainer.model = model
            metrics = trainer.evaluate(eval_dataset=eval_dataset)
            trainer.model.delete_adapter(trainer.adapter_name)
        else:
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        if debias_args.peft != 'sft' and debias_args.debias_configuration == 'after':
            model = trainer.model
            model.load_adapter(trainer.adapter_path)
            model.set_active_adapters(trainer.adapter_name)
            model.to(model.device)
            trainer.model = model
            predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
            trainer.model.delete_adapter(trainer.adapter_name)
        else:
            predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
        

        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = max_predict_samples
        
        if data_args.save_prefix == '':
            pref = 'predict'
        else:
            pref = f'predict_{data_args.save_prefix}'

        trainer.log_metrics(pref, metrics)
        trainer.save_metrics(pref, metrics)

        predictions = np.argmax(predictions, axis=1)
        if data_args.save_prefix == '':
            pred_output_path = 'predictions.txt'
        else:
            pred_output_path = f'predictions_{data_args.save_prefix}.txt'

        output_predict_file = os.path.join(training_args.output_dir, pred_output_path)
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    item = label_list[item]
                    writer.write(f"{index}\t{item}\n")

if __name__ == "__main__":
    main()

import json
import logging
import math
import random

from typing import Dict, Optional

import numpy as np
import torch

import datasets

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

BATCH_SOURCE_KEY = '_source'


def load_single_dataset(
    dataset_json,
    training_args,
    source=None,
    split=None,
    preprocessor=None,
    provide_source_to_preprocessor=None,
    cache_dir=None,
    max_seq_length=None,
    preprocessing_num_workers=None,
    overwrite_cache=False,
    remove_original_columns=False,
):
    split_map = {
        s: dataset_json.get(f'{s}_split', s)
        for s in ['train', 'validation', 'test']
    }

    load_kwargs = dataset_json.get('load_kwargs', {})

    data_files = {}
    if 'train_file' in dataset_json:
        data_files['train'] = dataset_json['train_file']
    if 'validation_file' in dataset_json:
        data_files['validation'] = dataset_json['validation_file']
    if 'test_file' in dataset_json:
        data_files['test'] = dataset_json['test_file']
    
    if 'name' in dataset_json:
        kwargs = {}
        if 'config_name' in dataset_json:
            load_kwargs['name'] = dataset_json['config_name']
        if data_files:
            load_kwargs['data_files'] = data_files

        raw_datasets = datasets.load_dataset(
            dataset_json['name'],
            cache_dir=cache_dir,
            **load_kwargs,
        )
    else:
        if 'file_type' in dataset_json:
            file_type = dataset_json['file_type']
        else:
            file_name = list(data_files.values())[0]
            file_type = file_name.split('.')[-1]
        if file_type.lower() == 'tsv':
            file_type = 'csv'
            load_kwargs['delimiter'] = '\t'
        raw_datasets = datasets.load_dataset(
            file_type,
            data_files=data_files,
            cache_dir=cache_dir,
            **load_kwargs
        )

    canonical_datasets = {}
    for canonical_split_name, names_in_dataset in split_map.items():
        if names_in_dataset is None:
            continue

        if not isinstance(names_in_dataset, list):
            names_in_dataset = [names_in_dataset]

        component_datasets = []
        for name in names_in_dataset:
            if name in raw_datasets:
                component_datasets.append(raw_datasets[name])
            elif name != canonical_split_name:
                raise ValueError(f'Dataset contains no split "{name}"')
        if len(component_datasets) == 0:
            continue

        canonical_datasets[canonical_split_name] = datasets.concatenate_datasets(
            component_datasets
        )

    if split is not None:
        if split not in canonical_datasets:
            return {}

        canonical_datasets = {split: canonical_datasets[split]}

    max_samples_by_split = {}
    if 'max_train_samples' in dataset_json:
        max_samples_by_split['train'] = int(dataset_json['max_train_samples'])
    if 'max_eval_samples' in dataset_json:
        max_samples_by_split['validation'] = int(dataset_json['max_eval_samples'])

    for key, dataset in canonical_datasets.items():
        max_samples = max_samples_by_split.get(key, None)
        if max_samples is not None:
            dataset = dataset.select(range(max_samples))

        if preprocessor is not None:
            active_preprocessor = preprocessor
            if isinstance(preprocessor, dict):
                active_preprocessor = preprocessor[key]
            if remove_original_columns:
                remove_columns = dataset.column_names
            else:
                remove_columns = []

            fn_kwargs = {}
            if provide_source_to_preprocessor:
                fn_kwargs['source'] = source

            with training_args.main_process_first(desc='dataset map pre-processing'):
                dataset = dataset.map(
                    active_preprocessor,
                    batched=True,
                    num_proc=preprocessing_num_workers,
                    remove_columns=remove_columns,
                    load_from_cache_file=not overwrite_cache,
                    desc='Preprocessing dataset',
                    fn_kwargs=fn_kwargs,
                )

            if max_seq_length is not None:
                dataset = dataset.filter(
                    lambda example: len(example['input_ids']) <= max_seq_length
                )

        canonical_datasets[key] = dataset

    return canonical_datasets
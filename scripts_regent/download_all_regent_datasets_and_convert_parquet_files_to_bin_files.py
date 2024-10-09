#!/usr/bin/env python3

import argparse
import os
from datasets import get_dataset_config_names, load_dataset
from datasets.config import HF_DATASETS_CACHE
from regent.eval.rl.core import SEEN_TASK_NAME_TO_ENV_ID, UNSEEN_TASK_NAME_TO_ENV_ID, TASK_NAME_TO_ENV_ID
from regent.utils import get_optional_suffix
import numpy as np
import json

## download all datasets of parquet files from huggingface
parser = argparse.ArgumentParser()
parser.add_argument("--tasks", nargs="+", default=["all"])

tasks = parser.parse_args().tasks
if tasks == ["all"]:
    tasks = ["atari", "babyai", "metaworld", "mujoco"]

for domain in ["atari", "babyai", "metaworld", "mujoco"]:
    if domain in tasks:
        tasks.remove(domain)
        tasks.extend([env_id for env_id in TASK_NAME_TO_ENV_ID.keys() if env_id.startswith(domain)])

for task in tasks:
    print(f"\nLoading {task}...")
    ## local save location for the bin files
    atari_dist_type = f'resnet18_512'
    finetune_num_demos = None
    optional_suffix = get_optional_suffix(task, atari_dist_type, finetune_num_demos)
    save_dir = f"{HF_DATASETS_CACHE}/regent-research/regent-subset-of-jat-dataset-tokenized-local/{task}{optional_suffix}"
    os.makedirs(f"{HF_DATASETS_CACHE}/regent-research/regent-subset-of-jat-dataset-tokenized-local", exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    ## check if the bin files already exist
    if not os.path.exists(f'{save_dir}/shapes.json'):
        ## if not, load remote dataset of parquet files, 
        ## and load their train split; there is no test split
        if task in SEEN_TASK_NAME_TO_ENV_ID:
            subset = load_dataset("regent-research/regent-subset-of-jat-dataset-tokenized", f'{task}_subset')['train']
        newdata = load_dataset("regent-research/regent-subset-of-jat-dataset-tokenized", f'{task}_newdata')['train']

        ## get each column (key) as a separate bin file
        ## keep track of all shapes in a dictionary
        all_shapes = {}
        if task in SEEN_TASK_NAME_TO_ENV_ID:
            for key in subset.column_names:
                if key != 'image_observations': # no need to save image observations; they are on huggingface for visualization only; we use the embeddings only in REGENT
                    assert type(subset[key]) == list
                    temp_array = np.array(subset[key]).astype(np.int32 if key in ['discrete_actions', 'discrete_observations'] else np.float32)
                    ### store shape
                    all_shapes[key] = temp_array.shape
                    ### save
                    print(f'--> saving {key} array of shape {temp_array.shape} to {save_dir}/{key}.bin')
                    temp_array.tofile(f'{save_dir}/{key}.bin')
        
        for key in newdata.column_names:
            assert type(newdata[key]) == list
            temp_array = np.array(newdata[key]).astype(np.int32 if key in ['indices'] else np.float32)
            ### store shape
            all_shapes[key] = temp_array.shape
            ### save
            print(f'--> saving {key} array of shape {temp_array.shape} to {save_dir}/{key}.bin')
            temp_array.tofile(f'{save_dir}/{key}.bin')

        ## save the shapes dictionary
        print(f'--> saving shapes dictionary to {save_dir}/shapes.json')
        with open(f"{save_dir}/shapes.json", 'w') as f:
            json.dump(all_shapes, f)
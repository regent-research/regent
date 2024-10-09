#!/usr/bin/env python3
"""Train a model from-srcatch on finetuning demos."""


import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import datasets.config
from datasets import load_dataset, load_from_disk
from datasets.config import HF_DATASETS_CACHE, HF_DATASETS_OFFLINE
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, HfArgumentParser, Trainer, TrainingArguments

from regent.eval.rl.core import SEEN_TASK_NAME_TO_ENV_ID, UNSEEN_TASK_NAME_TO_ENV_ID
from regent.utils import get_all_row_idxs_for_num_demos, get_all_row_idxs_for_100k_states
from regent.atari_utils import _LIMITED_ACTION_SET
import torch
from regent.modeling_from_scratch import ImpalaCNN, BC_MLP
import numpy as np


# Sometimes, the server is down; increasing the number of
# retries allows to wait more instead of making the training crash
datasets.config.STREAMING_READ_MAX_RETRIES = 10000


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config we are going to train from.
    """
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it "
                "will execute code present on the Hub on your local machine."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    tasks: List[str] = field(default_factory=list, metadata={"help": "Tasks to train on."})
    finetune_num_demos: int = field(default=None, metadata={"help": "Number of episodes (aka demos) to retrieve from and finetune on."})


SAMPLE_WEIGHTS = {
    "conceptual-captions": 10.0,
    "oscar": 10.0,
    "wikipedia": 10.0,
}

os.environ["WANDB_ENTITY"] = "regent-creators"
os.environ["WANDB_PROJECT"] = "from_scratch"


class MyTrainer(Trainer):
    def _get_train_sampler(self) -> None:
        return None
    
class TrainFromScratchDataset(torch.utils.data.Dataset):
    def __init__(self, train_dataset):
        self.train_dataset = train_dataset
        self.all_indices = []
        self.all_keys = {}
        for task, dataset in train_dataset.items():
            attn_masks = np.array(dataset['attention_mask']).astype(bool)
            B = 32 if task.startswith("atari") else 256
            assert attn_masks.shape == (len(dataset), B)
            keys = list(dataset[0].keys())
            self.all_keys[task] = keys
            for i in tqdm(range(len(dataset))):
                for j in range(B):
                    if attn_masks[i, j]:
                        self.all_indices.append((task, i, j))
        print(f'finished dataset setup: {self.all_keys=}')

    def __len__(self):
        return len(self.all_indices)

    def __getitem__(self, idx):
        task, i, j = self.all_indices[idx]
        keys = self.all_keys[task]
        datarow = self.train_dataset[task][i]
        outputs = {key: np.array(datarow[key][j]).astype(np.int32 if key in ['discrete_actions', 'discrete_observations'] else np.float32) for key in keys}
        for key in keys:
            outputs[key] = torch.from_numpy(outputs[key])
            if key in ['discrete_actions', 'discrete_observations']:
                outputs[key] = outputs[key].long()
        return outputs


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    # Set the tasks
    tasks = data_args.tasks
    if tasks == ["all"]:
        tasks = ["atari", "babyai", "metaworld", "mujoco", "oscar", "wikipedia", "conceptual-captions", "ok-vqa"]

    for domain in ["atari", "babyai", "metaworld", "mujoco"]:
        if domain in tasks:
            tasks.remove(domain)
            tasks.extend([env_id for env_id in SEEN_TASK_NAME_TO_ENV_ID.keys() if env_id.startswith(domain)])

    assert len(tasks) == 1, f'Expected 1 task, got {len(tasks)} tasks: {tasks}'

    # Setup from-scratch model
    if tasks[0].startswith("atari"):
        model = ImpalaCNN(task=tasks[0], shape=(4, 84, 84), num_actions=len(_LIMITED_ACTION_SET))
    else:
        model = BC_MLP(task=tasks[0], hidden_dims=[256, 256])

    # print number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params/ 1e6:.4f}M")

    # Load the datasets
    if HF_DATASETS_OFFLINE:
        for task in tasks:
            if not os.path.exists(f"{HF_DATASETS_CACHE}/jat-project/jat-dataset-tokenized/{task}"):
                raise ValueError(
                    f"""Dataset {task} not found in {HF_DATASETS_CACHE}/jat-project/jat-dataset-tokenized/
Make sure to download and save it first with
```
from datasets import load_dataset
dataset = load_dataset('jat-project/jat-dataset-tokenized', '{task}')
dataset.save_to_disk('{HF_DATASETS_CACHE}/jat-project/jat-dataset-tokenized/{task}')
```"""
                )
        
        train_dataset = {}
        for task in tqdm(tasks, desc="Loading datasets"):
            print(f'Loading {task} dataset')
            d = load_from_disk(f"{HF_DATASETS_CACHE}/jat-project/jat-dataset-tokenized/{task}")
            train_dataset[task] = d["train"]
            
            if (data_args.finetune_num_demos is not None and task in UNSEEN_TASK_NAME_TO_ENV_ID):
                train_dataset[task] = train_dataset[task].select(get_all_row_idxs_for_num_demos(task, data_args.finetune_num_demos))
    else:
        raise NotImplementedError("Online datasets are not supported yet.")

    train_dataset = TrainFromScratchDataset(train_dataset)
    print(f'------------------- len(train_dataset) = {len(train_dataset)} -------------------')

    # Due to the train dataset's structure, where every 'n' consecutive samples share the same modalities, we can't
    # load all samples at once. Different sets of 'n' samples have different modalities. Therefore, we must load and
    # process each set of 'n' samples separately.
    if training_args.dispatch_batches is not False:
        raise ValueError("Make sure to pass `--dispatch_batches False`.")

    # Why the training continue after exauhsting the dataset? https://github.com/huggingface/transformers/issues/26635
    trainer = MyTrainer(model=model, args=training_args, train_dataset=train_dataset)
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    if data_args.finetune_num_demos is not None:
        # save final checkpoint when finetuning
        os.makedirs(f"{training_args.output_dir}/checkpoint-final/", exist_ok=True)
        trainer.save_model(f"{training_args.output_dir}/checkpoint-final/")


if __name__ == "__main__":
    main()

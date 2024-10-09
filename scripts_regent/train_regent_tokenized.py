#!/usr/bin/env python3
"""Train a REGENT model"""


import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import torch
from datetime import datetime

import datasets.config
from datasets import load_dataset, load_from_disk, IterableDataset
from datasets.config import HF_DATASETS_CACHE, HF_DATASETS_OFFLINE
from tqdm import tqdm
from copy import deepcopy
from transformers import AutoConfig, AutoProcessor, HfArgumentParser, Trainer, TrainingArguments, TrainerCallback

from jat.eval.rl.core import TASK_NAME_TO_ENV_ID
from jat.utils_interleave_datasets import interleave_datasets
from regent.eval.rl.core import UNSEEN_TASK_NAME_TO_ENV_ID, SEEN_TASK_NAME_TO_ENV_ID
from regent.modeling_regent import JatRegentModel
from regent.utils import myprint
from scripts_regent.new_dataset import RetrievalAugmentedDataset, CombinedRetrievalAugmentedDataset


# Sometimes, the server is down; increasing the number of
# retries allows to wait more instead of making the training crash
datasets.config.STREAMING_READ_MAX_RETRIES = 10000


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config we are going to train from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
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
    lamda: float = field(default=10.0, metadata={"help": "lamda for the action interpolation."})
    use_global_atari_actions: bool = field(default=True, metadata={"help": "Whether to use global atari actions during training (note: data and env use local actions)."})
    
    mujoco_dist_multiplier: float = field(default=1.0, metadata={"help": "Redundant"})
    atari_dist_multiplier: float = field(default=1.0, metadata={"help": "Redundant"})

    dist_normalizer: str = field(default="p95", metadata={"help": "Normalize distances by one of [std, p80, p85, p90, p95, p99]."})

    atari_dist_type: str = field(default='resnet18_512', metadata={"help": "Type of distance to use for atari retrieval"})

    use_atari_embeddings: bool = field(default=True, metadata={"help": "Whether to use Atari embeddings instead of images."})


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
os.environ["WANDB_PROJECT"] = "jat_regent"


class MyTrainer(Trainer):
    def _get_train_sampler(self) -> None:
        return None # stops random sampling in dataloader!
    
class CustomSaveCallback(TrainerCallback):
    """Callback to save after every epoch along with chosen save_steps."""
    def on_epoch_end(self, args, state, control, **kwargs):
        control.should_save = True
        return control


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    myprint(f'{model_args=}')
    myprint(f'{data_args=}')
    myprint(f'{training_args=}')

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    # task divisions and check if only RL tasks are given
    RL_tasks = ["atari", "babyai", "metaworld", "mujoco"]
    text_tasks = ["oscar", "wikipedia"]
    vqa_tasks = ["conceptual-captions", "ok-vqa"]

    ONLY_RL_TASKS = True
    OG_tasks = deepcopy(data_args.tasks)
    for task in OG_tasks:
        if task in text_tasks + vqa_tasks:
            ONLY_RL_TASKS = False
            break

    # model
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    config.lamda = model_args.lamda
    config.use_global_atari_actions = model_args.use_global_atari_actions
    config.mujoco_dist_multiplier = model_args.mujoco_dist_multiplier
    config.atari_dist_multiplier = model_args.atari_dist_multiplier
    config.dist_normalizer = model_args.dist_normalizer
    config.atari_dist_type = model_args.atari_dist_type
    config.use_atari_embeddings = model_args.use_atari_embeddings
    config.finetune_num_demos = data_args.finetune_num_demos
    model = JatRegentModel(config)
    processor = AutoProcessor.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )

    # load state_dict if finetuning; also change wandb project
    if config.finetune_num_demos is not None:
        state_dict_loc = f"{model_args.model_name_or_path}/pytorch_model.bin"
        model.load_state_dict(torch.load(state_dict_loc))
        myprint(f'loaded state_dict from {state_dict_loc}')
        os.environ["WANDB_PROJECT"] = "jat_regent_finetune"

    # Set the tasks
    tasks = data_args.tasks
    if tasks == ["all"]:
        tasks = RL_tasks + text_tasks + vqa_tasks
                
    for domain in ["atari", "babyai", "metaworld", "mujoco"]:
        if domain in tasks:
            tasks.remove(domain)
            tasks.extend([env_id for env_id in SEEN_TASK_NAME_TO_ENV_ID.keys() if env_id.startswith(domain)])

    # Load the datasets
    if HF_DATASETS_OFFLINE:
        # train hf dataset which is already shuffled!
        train_dataset = CombinedRetrievalAugmentedDataset(tasks, ONLY_RL_TASKS, n_contiguous=training_args.per_device_train_batch_size, split="train", num_contexts=config.num_contexts, 
                                                          lamda=config.lamda, use_global_atari_actions=config.use_global_atari_actions, dist_multipliers={"mujoco": config.mujoco_dist_multiplier, "atari": config.atari_dist_multiplier},
                                                          dist_normalizer=config.dist_normalizer, atari_dist_type=config.atari_dist_type, use_atari_embeddings=config.use_atari_embeddings, 
                                                          finetune_num_demos=config.finetune_num_demos if task in UNSEEN_TASK_NAME_TO_ENV_ID else None)
        myprint(f'------------------- len(train_dataset) = {len(train_dataset)} -------------------')

        # eval hf dataset (not shuffled)
        all_eval_datasets = {}
        for domain in OG_tasks:
            curr_tasks = [env_id for env_id in SEEN_TASK_NAME_TO_ENV_ID.keys() if env_id.startswith(domain)]
            eval_dataset = CombinedRetrievalAugmentedDataset(curr_tasks, ONLY_RL_TASKS, n_contiguous=training_args.per_device_train_batch_size, split="eval", num_contexts=config.num_contexts, 
                                                             lamda=config.lamda, use_global_atari_actions=config.use_global_atari_actions, dist_multipliers={"mujoco": config.mujoco_dist_multiplier, "atari": config.atari_dist_multiplier},
                                                             dist_normalizer=config.dist_normalizer, atari_dist_type=config.atari_dist_type, use_atari_embeddings=config.use_atari_embeddings, 
                                                             finetune_num_demos=config.finetune_num_demos if task in UNSEEN_TASK_NAME_TO_ENV_ID else None)
            all_eval_datasets[domain] = eval_dataset
            myprint(f'------------------- {domain} | len(eval_dataset) = {len(eval_dataset)} -------------------')
        if config.finetune_num_demos is not None:
            all_eval_datasets = None
    else:
        raise NotImplementedError("Online datasets are not supported yet.")

    # Due to the train dataset's structure, where every 'n' consecutive samples share the same modalities, we can't
    # load all samples at once. Different sets of 'n' samples have different modalities. Therefore, we must load and
    # process each set of 'n' samples separately.
    if training_args.dispatch_batches is not False:
        raise ValueError("Make sure to pass `--dispatch_batches False`.")

    # Why the training continue after exauhsting the dataset? https://github.com/huggingface/transformers/issues/26635
    trainer = MyTrainer(model=model, args=training_args, train_dataset=train_dataset, tokenizer=processor, eval_dataset=all_eval_datasets, callbacks=[CustomSaveCallback()])
    if config.finetune_num_demos is None:
        trainer.evaluate()
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    if config.finetune_num_demos is not None:
        # save final checkpoint when finetuning
        os.makedirs(f"{training_args.output_dir}/checkpoint-final/", exist_ok=True)
        trainer.save_model(f"{training_args.output_dir}/checkpoint-final/")

if __name__ == "__main__":
    main()

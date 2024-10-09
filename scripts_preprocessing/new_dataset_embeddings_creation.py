import numpy as np 
from datetime import datetime
from datasets import load_from_disk, Dataset, concatenate_datasets
from datasets.config import HF_DATASETS_CACHE
from regent.utils import myprint, get_task_info, load_retrieved_indices, process_row_of_obs_atari_full_without_mask, get_emb_transform_model_dim, get_optional_suffix
from dataclasses import dataclass, field
from typing import List
from transformers import HfArgumentParser
from copy import deepcopy
from jat.eval.rl import TASK_NAME_TO_ENV_ID
from regent.eval.rl import SEEN_TASK_NAME_TO_ENV_ID, UNSEEN_TASK_NAME_TO_ENV_ID
from pytorch_msssim import ssim
import torch
import os
import json
import logging


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config we are going to train from.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to output folder"}
    )
    atari_dist_type_OR: str = field(default='resnet18_512', metadata={"help": "Type of distance to use for loading retrieved indices"})
    atari_dist_type_emb: str = field(default='resnet18_512', metadata={"help": "Type of distance to use for computing the embeddings and saving them"})


@dataclass
class EvaluationArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    tasks: List[str] = field(default_factory=list, metadata={"help": "Tasks to train on."})
    num_contexts: int = field(default=20, metadata={"help": "Number of states in context (includes 1 query state)."})
    finetune_num_demos: int = field(default=None, metadata={"help": "Number of episodes (aka demos) to retrieve from and finetune on."})


def main():
    parser = HfArgumentParser((ModelArguments, EvaluationArguments))
    model_args, eval_args = parser.parse_args_into_dataclasses()
    atari_dist_type_OR = model_args.atari_dist_type_OR
    atari_dist_type_emb = model_args.atari_dist_type_emb
    finetune_num_demos = eval_args.finetune_num_demos

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.WARNING, #logging.INFO, # chnaged this so that index creation logs are not printed!
    )

    # Set the tasks
    OG_tasks = deepcopy(eval_args.tasks)
    tasks = eval_args.tasks
    for domain in ["atari", "babyai", "metaworld", "mujoco"]:
        if domain in tasks:
            tasks.remove(domain)
            tasks.extend([env_id for env_id in TASK_NAME_TO_ENV_ID.keys() if env_id.startswith(domain)])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # iterate over tasks
    myprint(f'all tasks: {tasks}')
    for task in tasks:
        myprint(('-'*100) + f'{task=}')
        
        # optional_suffix and save_loc
        optional_suffix = get_optional_suffix(task, atari_dist_type_OR, finetune_num_demos)
        save_loc = f"{HF_DATASETS_CACHE}/regent-research/regent-subset-of-jat-dataset-tokenized-local/{task}{optional_suffix}"
        os.makedirs(save_loc, exist_ok=True)
        # continue to next task if already done
        if os.path.exists(f'{save_loc}/embeddings_{atari_dist_type_emb}.bin'):
            myprint(f'<{save_loc}/embeddings_{atari_dist_type_emb}.bin already exists!>')
            myprint(f'<{task} already done!>')
            continue

        # load dataset
        dataset = load_from_disk(f"{HF_DATASETS_CACHE}/jat-project/jat-dataset-tokenized/{task}")
        rew_key, attn_key, obs_key, act_key, B, obs_dim, act_dim = get_task_info(task)
        extra_key = 'discrete_RandP_action_logits' if task.startswith("atari") or task.startswith("babyai") else 'continuous_RandP_actions'

        # get embedding model
        if task.startswith("atari"):
            emb_transform, emb_model, emb_dim = get_emb_transform_model_dim(atari_dist_type_emb, device)
            obs_dim = emb_dim # overwrite for atari_dist_type

        # load the retrieved_indices
        local_path = f'dataset_jat_regent/{task}'
        myprint(f'<loading retrieved_indices>')
        retrieved_indices_dict = load_retrieved_indices(task, local_path, atari_dist_type=atari_dist_type_OR, finetune_num_demos=finetune_num_demos)
        myprint(f'</loading retrieved_indices>')
        
        # load row idxs for seen and unseen tasks
        all_row_idxs = list(retrieved_indices_dict.keys())
        myprint(f'{len(all_row_idxs)=}')

        # compute distances and save indices
        myprint(f'<computing embeddings>')
        all_embeddings = []
        for count, row_idx in enumerate(all_row_idxs):
            myprint(f'working on {row_idx=}: {count}/{len(retrieved_indices_dict)}')
            
            states = dataset['train'][row_idx][obs_key]
            states = process_row_of_obs_atari_full_without_mask(states)
            states = torch.from_numpy(states).to(device)
            with torch.no_grad():
                states = emb_model(emb_transform(states)).cpu().numpy()
            assert states.shape == (B, *emb_dim), f'{states.shape=}, {(B, *emb_dim)=}'

            all_embeddings.append(states)
        all_embeddings = np.stack(all_embeddings).astype(np.float32)
        assert all_embeddings.shape == (len(all_row_idxs), B, *emb_dim), f'{all_embeddings.shape=}, {(len(all_row_idxs), B, *emb_dim)=}'

        # save
        all_embeddings.tofile(f"{save_loc}/embeddings_{atari_dist_type_emb}.bin")
        myprint(f'saved to {save_loc}/embeddings_{atari_dist_type_emb}.bin')
                    
if __name__ == "__main__":
    main()
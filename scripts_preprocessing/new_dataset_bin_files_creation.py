import numpy as np 
from datetime import datetime
from datasets import load_from_disk, Dataset, concatenate_datasets
from datasets.config import HF_DATASETS_CACHE
from regent.utils import myprint, get_task_info, load_retrieved_indices, process_row_of_obs_atari_full_without_mask, L2dist, get_emb_transform_model_dim, get_optional_suffix
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
    atari_dist_type: str = field(default='resnet18_512', metadata={"help": "Type of distance to use for atari retrieval"})


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
    atari_dist_type = model_args.atari_dist_type
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
        optional_suffix = get_optional_suffix(task, atari_dist_type, finetune_num_demos)
        save_loc = f"{HF_DATASETS_CACHE}/regent-research/regent-subset-of-jat-dataset-tokenized-local/{task}{optional_suffix}"
        os.makedirs(save_loc, exist_ok=True)
        # continue to next task if already done
        if os.path.exists(f'{save_loc}/shapes.json'):
            myprint(f'<{save_loc}/shapes.json exists>')
            myprint(f'<{task} already done!>')
            continue

        # load dataset
        dataset = load_from_disk(f"{HF_DATASETS_CACHE}/jat-project/jat-dataset-tokenized/{task}")
        rew_key, attn_key, obs_key, act_key, B, obs_dim, act_dim = get_task_info(task)
        extra_key = 'discrete_RandP_action_logits' if task.startswith("atari") or task.startswith("babyai") else 'continuous_RandP_actions'

        # get embedding model
        if task.startswith("atari"):
            emb_transform, emb_model, emb_dim = get_emb_transform_model_dim(atari_dist_type, device)
            obs_dim = emb_dim # overwrite for atari_dist_type

        # load the retrieved_indices
        local_path = f'dataset_jat_regent/{task}'
        myprint(f'<loading retrieved_indices>')
        retrieved_indices_dict = load_retrieved_indices(task, local_path, atari_dist_type=atari_dist_type, finetune_num_demos=finetune_num_demos)
        myprint(f'</loading retrieved_indices>')
        
        # load row idxs for seen and unseen tasks
        all_row_idxs = list(retrieved_indices_dict.keys())
        myprint(f'{len(all_row_idxs)=}')

        # load dataset_rows, attn_masks
        myprint(f'<loading dataset required rows and attn masks>')
        dataset_rows = {row_idx: dataset['train'][row_idx] for row_idx in all_row_idxs}
        attn_masks = {}
        for row_idx in all_row_idxs:
            attn_masks[row_idx] = np.array(dataset_rows[row_idx][attn_key]).astype(bool)
            assert attn_masks[row_idx].shape == (B,)
        myprint(f'</loading dataset required rows and attn masks>')

        # compute distances and save indices
        myprint(f'<computing new features: distances, indices>')
        new_features = {k: [] for k in ['distances', 'indices']}
        for count, row_idx in enumerate(all_row_idxs):
            myprint(f'working on {row_idx=}: {count}/{len(retrieved_indices_dict)}')
            for col_idx in range(B):
                if attn_masks[row_idx][col_idx]:
                    # append an empty list to new_features for each key
                    for k in ['distances', 'indices']:
                        new_features[k].append([])

                    # get current retrieved indices
                    curr_retrieved_indices = retrieved_indices_dict[row_idx][col_idx][:eval_args.num_contexts - 1]

                    # load retrieved indices into new_features; save states for distance computation
                    states = []
                    for rr, rc in curr_retrieved_indices:
                        assert attn_masks[rr][rc]
                        new_features['indices'][-1].append([rr, rc])
                        states.append(dataset_rows[rr][obs_key][rc])
                    
                    rr0, rc0 = curr_retrieved_indices[0]

                    # load query indices into new_features; save states for distance computation
                    new_features['indices'][-1].append([row_idx, col_idx])
                    assert len(new_features['indices'][-1]) == eval_args.num_contexts
                    states.append(dataset_rows[row_idx][obs_key][col_idx])

                    # process states for distance computation
                    if task.startswith("atari"):
                        states = process_row_of_obs_atari_full_without_mask(states)
                        states = torch.from_numpy(states).to(device)
                        with torch.no_grad():
                            states = emb_model(emb_transform(states)).cpu().numpy()
                    elif task.startswith("babyai"):
                        states = np.array(states)[:, :148] # removing last 64 text tokens
                    else:
                        states = np.array(states)
                    assert states.shape == (eval_args.num_contexts, *obs_dim)

                    # compute distances
                    first_state = states[0:1]
                    assert first_state.shape == (1, *obs_dim)
                    new_features['distances'][-1].append(0.0)
                    for i in range(1, eval_args.num_contexts):
                        curr_state = states[i:i+1]                        
                        assert curr_state.shape == (1, *obs_dim)

                        dist = L2dist(first_state, curr_state)
                        new_features['distances'][-1].append(dist)
                    assert len(new_features['distances'][-1]) == eval_args.num_contexts

        # collect all data to save as bin files and their shapes
        myprint(f'<collecting all data to save as bin files>')
        all_og_keys = [obs_key, act_key, rew_key] #list(dataset['train'].features.keys()) # no need attn_key or loss_weight!
        all_data = {k: np.array([dataset_rows[r][k] for r in all_row_idxs]).astype(np.int32 if k in ['discrete_actions', 'discrete_observations'] else np.float32)
                    for k in all_og_keys}
        all_data['distances'] = np.array(new_features['distances']).astype(np.float32)
        all_data['indices'] = np.array(new_features['indices']).astype(np.int32)

        all_shapes = {k: all_data[k].shape for k in all_data.keys()}
        myprint(f'all_shapes={all_shapes}')

        # save all tofile
        for k in all_data.keys():
            all_data[k].tofile(f"{save_loc}/{k}.bin")
            myprint(f'saved to {save_loc}/{k}.bin')
        with open(f"{save_loc}/shapes.json", 'w') as f:
            json.dump(all_shapes, f)
        myprint(f'saved to {save_loc}/shapes.json')

        # save a map between actal row_idx in all_row_idxs and its location in all_row_idxs
        all_row_idxs_lookup = {r: i for i, r in enumerate(all_row_idxs)}
        save_loc_2 = f'{local_path}/all_row_idxs_lookup{optional_suffix}.json'
        with open(save_loc_2, 'w') as f:
            json.dump(all_row_idxs_lookup, f)
        myprint(f'saved to {save_loc_2}')

if __name__ == "__main__":
    main()
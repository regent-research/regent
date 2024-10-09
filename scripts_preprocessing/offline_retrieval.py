import numpy as np 
import torch 
import os
import sys
import json
from datasets import load_from_disk
from datasets.config import HF_DATASETS_CACHE
from regent.utils import (myprint, get_task_info, collect_all_data, retrieve_vector, 
                              build_index_vector, get_all_row_idxs_for_100k_states, process_row_of_obs_atari_full_without_mask,
                              get_num_demos, get_emb_transform_model_dim
                            )
from dataclasses import dataclass, field
from typing import List
from transformers import AutoProcessor, HfArgumentParser
from copy import deepcopy
from jat.eval.rl import TASK_NAME_TO_ENV_ID
from regent.eval.rl import SEEN_TASK_NAME_TO_ENV_ID, UNSEEN_TASK_NAME_TO_ENV_ID
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
    use_cpu: bool = field(default=False, metadata={"help": "Use CPU instead of GPU."})
    finetune_num_demos: int = field(default=None, metadata={"help": "Number of episodes (aka demos) to retrieve from and finetune on."})


def get_default_device() -> torch.device:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def main():
    parser = HfArgumentParser((ModelArguments, EvaluationArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, eval_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, eval_args = parser.parse_args_into_dataclasses()
    atari_dist_type = model_args.atari_dist_type

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Set the tasks
    OG_tasks = deepcopy(eval_args.tasks)
    tasks = eval_args.tasks
    for domain in ["atari", "babyai", "metaworld", "mujoco"]:
        if domain in tasks:
            tasks.remove(domain)
            tasks.extend([env_id for env_id in TASK_NAME_TO_ENV_ID.keys() if env_id.startswith(domain)])

    device = torch.device("cpu") if eval_args.use_cpu else get_default_device()

    # iterate over tasks
    for task in tasks:
        myprint(('-'*100) + f'{task=}')
        dataset = load_from_disk(f"{HF_DATASETS_CACHE}/jat-project/jat-dataset-tokenized/{task}")

        # more args
        batch_size_retrieval = 16
        nb_cores_autofaiss = 8 # for other vector obs envs
        num_to_retrieve = 50 if task.startswith("babyai") else 100 # babyai sometimes has like 5 steps in an ep; so retrieving 100 from 20 eps is too much!

        # load from local_path
        local_path = f"dataset_jat_regent/{task}"
        with open(f"{local_path}/rows_tokenized_2_eps.json", 'r') as f:
            rows_tokenized_2_eps = json.load(f)
        rows_tokenized_2_eps = {int(k): v for k, v in rows_tokenized_2_eps.items()}
        with open(f"{local_path}/eps_2_rows_tokenized.json", 'r') as f:
            eps_2_rows_tokenized = json.load(f)
        eps_2_rows_tokenized = {int(k): v for k, v in eps_2_rows_tokenized.items()}
        if task.startswith("babyai"):
            with open(f"{local_path}/eps_2_text.json", 'r') as f:
                eps_2_text = json.load(f)
            eps_2_text = {int(k): v for k, v in eps_2_text.items()}

        # setup
        rew_key, attn_key, obs_key, act_key, B, obs_dim, act_dim = get_task_info(task)
        
        # get embedding model
        if task.startswith("atari"):
            emb_transform, emb_model, emb_dim = get_emb_transform_model_dim(atari_dist_type, device)
            obs_dim = emb_dim # overwrite for atari_dist_type
        
        kwargs = {'B': B,
                'obs_dim': obs_dim,
                'attn_key': attn_key,
                'obs_key': obs_key,
                'device': device,
                'task': task,
                'batch_size_retrieval': batch_size_retrieval,
                'nb_cores_autofaiss': nb_cores_autofaiss,
                'verbose': False,
                'atari_dist_type': atari_dist_type,
                }
        
        # for retrieval states
        num_demos = get_num_demos(task, eval_args.finetune_num_demos)
        if task.startswith("babyai"):
            all_rows_of_obs_retrieval_dict, all_attn_masks_retrieval_dict, all_row_idxs_retrieval_dict, _ = collect_all_data(dataset, task, obs_key, num_demos=num_demos, atari_dist_type=atari_dist_type)
        else:
            all_rows_of_obs_retrieval, all_attn_masks_retrieval, all_row_idxs_retrieval, _ = collect_all_data(dataset, task, obs_key, num_demos=num_demos, atari_dist_type=atari_dist_type)

        # for query states
        if task.startswith("babyai"):
            all_row_idxs_query_dict = get_all_row_idxs_for_100k_states(task)
            sorted_combined_OG = sorted(sum(list(all_row_idxs_query_dict.values()), []))
        else:
            # if unseen tasks, query and retrieval sets are the same; for seen tasks, use 100k query states
            all_row_idxs_query = all_row_idxs_retrieval if task in UNSEEN_TASK_NAME_TO_ENV_ID else get_all_row_idxs_for_100k_states(task)

        # assert that all_row_idxs_retrieval is subset of all_row_idxs_query
        if task.startswith("babyai"):
            for mission_name in all_row_idxs_query_dict.keys():
                assert set(all_row_idxs_retrieval_dict[mission_name]).issubset(set(all_row_idxs_query_dict[mission_name]))
        else:
            assert set(all_row_idxs_retrieval).issubset(set(all_row_idxs_query))

        # get all query episodes
        if task.startswith("babyai"):
            all_eps_query = []
            for mission_name in all_row_idxs_query_dict.keys():
                all_eps_query += list(set([rows_tokenized_2_eps[row_idx] for row_idx in all_row_idxs_query_dict[mission_name]]))
        else:
            all_eps_query = list(set([rows_tokenized_2_eps[row_idx] for row_idx in all_row_idxs_query]))

        # if json exists, task is already done!
        save_path = f"{local_path}/retrieved_indices_{num_demos}demos{f'_emb_{atari_dist_type}' if task.startswith('atari') else ''}.json"
        if os.path.exists(f"{save_path}"):
            myprint(f'<{save_path} exists>')
            myprint(f'<{task} already done!>')
            continue
        else:
            all_retrieved_indices = {}

        # assert that all_eps_query is in ascending order
        all_eps_query = sorted(all_eps_query)

        # loop over query episodes
        use_prev = False
        for count, current_ep_query in enumerate(all_eps_query):
            myprint(f'\n\n\n<current_ep_query: {current_ep_query}: count {count}/{len(all_eps_query)}>')
            if task.startswith("babyai"):
                current_mission = eps_2_text[current_ep_query]
                all_row_idxs_retrieval = all_row_idxs_retrieval_dict[current_mission]
                all_row_idxs_query = all_row_idxs_query_dict[current_mission]
                all_rows_of_obs_retrieval = all_rows_of_obs_retrieval_dict[current_mission]
                all_attn_masks_retrieval = all_attn_masks_retrieval_dict[current_mission]

            # get rows from all retrieval episodes but the current query episode (in case of overlap)
            all_other_row_idxs_retrieval = [row_idx for row_idx in all_row_idxs_retrieval if rows_tokenized_2_eps[row_idx] != current_ep_query]
            if len(all_eps_query) == 1: # if only one real episode, take first half for retrieval
                all_other_row_idxs_retrieval = all_row_idxs_retrieval[:len(all_row_idxs_retrieval)//2]

            # Note: we may have: all_row_idxs_retrieval = [299, 300, 301, 302, 303], all_other_row_idxs_retrieval = [301, 302, 303], so we need to take_subset by using [2, 3, 4] indices. Doing this conversion below:
            all_other_row_idxs_for_taking_subset = [all_row_idxs_retrieval.index(idx) for idx in all_other_row_idxs_retrieval]
            if (task.startswith("babyai") or # above applies for babyai
                (task.startswith("atari") and all_row_idxs_retrieval != list(range(max(all_row_idxs_retrieval)+1)))): # above also applies for certain atari envs with discontinuities in row indices; comes from taking only first 200 rows in each episode! 
                myprint(f'Note: all_other_row_idxs_for_taking_subset != all_other_row_idxs_retrieval')
            else:
                assert all_other_row_idxs_for_taking_subset == all_other_row_idxs_retrieval, f'former {all_other_row_idxs_for_taking_subset}, latter {all_other_row_idxs_retrieval}'

            # take subset
            all_other_rows_of_obs_retrieval = all_rows_of_obs_retrieval[all_other_row_idxs_for_taking_subset]            
            all_other_attn_masks_retrieval = all_attn_masks_retrieval[all_other_row_idxs_for_taking_subset]

            # for retrieval (continued),
            if all_other_row_idxs_retrieval != all_row_idxs_retrieval or not use_prev:
                # create index, collect subset of data that we can retrieve from
                all_indices, knn_index = build_index_vector(all_rows_of_obs_OG=all_other_rows_of_obs_retrieval,
                                                            all_attn_masks_OG=all_other_attn_masks_retrieval,
                                                            all_row_idxs=all_other_row_idxs_retrieval,
                                                            kwargs=kwargs)
                if all_other_row_idxs_retrieval == all_row_idxs_retrieval:
                    use_prev = True # from now on we will use those created above at this time. No need to do it again for states beyond the retrieval states
                    if (task.startswith("babyai") or # above applies for babyai
                        (task.startswith("atari") and all_row_idxs_retrieval != list(range(max(all_row_idxs_retrieval)+1)))): # above also applies for certain atari envs with discontinuities in row indices; comes from taking only first 200 rows in each episode! 
                        myprint(f'Note: all_other_row_idxs_for_taking_subset != all_row_idxs_retrieval')
                    else:
                        assert all_other_row_idxs_for_taking_subset == all_row_idxs_retrieval
            else:
                myprint(f'<No need for processing obs/making index and collecting indices (all np arrays)>')

            # identify rows from current episode
            rows_from_current_ep_query = [row_idx for row_idx in eps_2_rows_tokenized[current_ep_query] if row_idx in all_row_idxs_query] # if cdn helps skip those rows which may be in eps but arent in query states
            
            # loop over rows from current episode
            for row_count, row_idx in enumerate(rows_from_current_ep_query):
                myprint(f'<row_idx: {row_idx}: row_count {row_count}/{len(rows_from_current_ep_query)}>')

                # obtain row of obs
                datarow = dataset['train'][row_idx]
                attn_mask = np.array(datarow[attn_key]).astype(bool)
                if task.startswith("atari"):
                    row_of_obs = process_row_of_obs_atari_full_without_mask(datarow[obs_key])
                    row_of_obs = torch.from_numpy(row_of_obs).to(device)
                    with torch.no_grad():
                        row_of_obs = emb_model(emb_transform(row_of_obs)).cpu().numpy()
                else:
                    row_of_obs = np.array(datarow[obs_key]).astype(np.float32)
                row_of_obs = row_of_obs[attn_mask]

                if task.startswith("babyai"):
                    assert row_of_obs.shape == (np.sum(attn_mask), 212) and isinstance(row_of_obs, np.ndarray)
                    row_of_obs = row_of_obs[:, :148] # removing last 64 text tokens
                
                # assert row of obs shape and type
                assert row_of_obs.shape == (np.sum(attn_mask), *obs_dim)
                assert isinstance(row_of_obs, np.ndarray)
                myprint(f'row_of_obs.shape: {row_of_obs.shape}')

                # Retrieve indices
                retrieved_indices = retrieve_vector(row_of_obs=row_of_obs, 
                                                    knn_index=knn_index, 
                                                    all_indices=all_indices, 
                                                    num_to_retrieve=num_to_retrieve,
                                                    kwargs=kwargs)
                    
                # pad the above to expected B
                xbdim = row_of_obs.shape[0]
                if xbdim < B:
                    retrieved_indices = np.concatenate([retrieved_indices, np.zeros((B-xbdim, num_to_retrieve, 2), dtype=int)], axis=0)
                assert retrieved_indices.shape == (B, num_to_retrieve, 2)
                myprint(f'retrieved_indices.shape: {retrieved_indices.shape}')

                # collect retrieved indices
                all_retrieved_indices[row_idx] = {}
                for col_idx in range(B):
                    all_retrieved_indices[row_idx][col_idx] = retrieved_indices[col_idx].tolist()

                myprint(f'</row_idx: {row_idx}: row_count {row_count}/{len(rows_from_current_ep_query)}>')
            myprint(f'</current_ep_query: {current_ep_query}: count {count}/{len(all_eps_query)}>')

        # some asserts checking that the rows we iterated over (ie all_row_idxs_post) are same as the ones we should have iterated over (ie all_row_idxs_query); maybe not needed since we add if cdn in rows_from_current_ep_query.
        all_row_idxs_post = list(all_retrieved_indices.keys())
        if task.startswith("babyai"):
            assert all_row_idxs_post == sorted_combined_OG, f'{all_row_idxs_post=},\n{sorted_combined_OG=}'
        else:
            assert all_row_idxs_post == all_row_idxs_query, f'{all_row_idxs_post=},\n{all_row_idxs_query=}'

        # save as json
        with open(save_path, 'w') as f:
            json.dump(all_retrieved_indices, f)
        myprint(f'<saved to {save_path}>')

if __name__ == "__main__":
    main()
from regent.eval.rl.core import SEEN_TASK_NAME_TO_ENV_ID, UNSEEN_TASK_NAME_TO_ENV_ID, TASK_NAME_TO_ENV_ID
import os 
import numpy as np 
from datasets.config import HF_DATASETS_CACHE
from datasets import Dataset
from regent.utils import myprint, get_task_info, load_all_row_idxs_lookup, get_dist_stats, get_optional_suffix
import json
from tqdm import tqdm


def convert_to_int64_if_necessary(data_dict):
    for key, value in data_dict.items():
        if np.issubdtype(value.dtype, np.integer) and value.dtype == np.int32:
            data_dict[key] = value.astype(np.int64)
    return data_dict

def debug_data_shapes(data_dict, dict_name):
    for key, value in data_dict.items():
        print(f"{dict_name} Key: {key}, dtype: {value.dtype}, shape: {value.shape}, max_value: {np.max(value) if np.issubdtype(value.dtype, np.integer) else 'N/A'}")


def main():
    repo_id = f'regent-research/regent-subset-of-jat-dataset-tokenized'

    ## setup all tasks
    all_tasks = list(TASK_NAME_TO_ENV_ID.keys())
    ## In case you want to run multiples of this python script in parallel
    # all_tasks = list(SEEN_TASK_NAME_TO_ENV_ID.keys())
    # all_tasks = [task for task in all_tasks if task.startswith("atari")]
    # all_tasks = list(UNSEEN_TASK_NAME_TO_ENV_ID.keys()) # only pushing distances for unseen datasets for calculating values to normalize distances

    for task in tqdm(all_tasks):
        myprint(f'{task=}')
        # args
        finetune_num_demos = None
        atari_dist_type = 'resnet18_512'
        use_atari_embeddings = True
        num_contexts = 20

        # init
        optional_suffix = get_optional_suffix(task, atari_dist_type, finetune_num_demos)
        folder_loc = f"{HF_DATASETS_CACHE}/regent-research/regent-subset-of-jat-dataset-tokenized-local/{task}{optional_suffix}"
        # folder_loc = f"{HF_DATASETS_CACHE}/regent-project/jat-regent-dataset-tokenized/{task}{optional_suffix}" ########### DELETE TEMP
        rew_key, attn_key, obs_key, act_key, B, obs_dim, act_dim = get_task_info(task)
        extra_key = 'discrete_RandP_action_logits' if task.startswith("atari") or task.startswith("babyai") else 'continuous_RandP_actions'
        local_path = f'dataset_jat_regent/{task}'
        all_row_idxs_lookup = load_all_row_idxs_lookup(task, local_path, optional_suffix=optional_suffix)

        # load shapes and bin files
        assert os.path.exists(folder_loc), f'{folder_loc} does not exist'
        with open(f"{folder_loc}/shapes.json", 'r') as f:
            all_shapes = json.load(f)
        all_shapes = {k: tuple(v_list) for k, v_list in all_shapes.items() if k != 'max_dist'}
        if task.startswith("atari"):
            emb_key = f'embeddings_resnet18_512'
            all_shapes[emb_key] = (all_shapes[act_key][0], all_shapes[act_key][1], 512)
        all_data = {k: np.memmap(f"{folder_loc}/{k}.bin", dtype='int32' if k in ['indices', 'discrete_actions', 'discrete_observations'] else 'float32', mode='r', shape=all_shapes[k])
                    for k in [obs_key, act_key, rew_key, 'distances', 'indices'] + ([emb_key] if task.startswith("atari") else [])}

        # separate subset data and distances/indices data
        # To load the full array into memory, you can make a copy:
        subset_data = {k: np.array(all_data[k]) for k in [obs_key, act_key, rew_key] + ([emb_key] if task.startswith("atari") else [])}
        new_data = {k: np.array(all_data[k]) for k in ['distances', 'indices']}
        del all_data
        myprint(f'loaded')

        ###### information for the reader of this code on the shapes of data
        # and asserts of loaded data
        num_all_row_idxs = len(all_row_idxs_lookup) # the number of rows in the jat-dataset-tokenized that were used to create this dataset; each row has a maximum of B states.
        num_total = all_shapes['indices'][0] # the total number of states in the above rows; also the length of this dataset
        if task.startswith("atari"): # overwrite obs_dim because raw obs in atari are (4, 84, 84) and raw obs in babyai have 64 extra dim
            obs_dim = (4, 84, 84)
            emb_dim = (512,)
            assert subset_data[emb_key].shape == all_shapes[emb_key] == (num_all_row_idxs, B, *emb_dim)
        elif task.startswith("babyai"):
            obs_dim = (obs_dim[0]+64,)
        assert subset_data[obs_key].shape == all_shapes[obs_key] == (num_all_row_idxs, B, *obs_dim)
        assert ((act_key == 'continuous_actions' and subset_data[act_key].shape == all_shapes[act_key] == (num_all_row_idxs, B, act_dim)) or 
                (act_key == 'discrete_actions' and subset_data[act_key].shape == all_shapes[act_key] == (num_all_row_idxs, B)))
        assert subset_data[rew_key].shape == all_shapes[rew_key] == (num_all_row_idxs, B)
        assert new_data['distances'].shape == all_shapes['distances'] == (num_total, num_contexts)
        assert new_data['indices'].shape == all_shapes['indices'] == (num_total, num_contexts, 2)
        myprint(f'asserted')

        # only if seen atari, convert only the image observations to list
        if task.startswith("atari") and task in SEEN_TASK_NAME_TO_ENV_ID:
            subset_data[obs_key] = subset_data[obs_key].tolist()
            myprint(f'tolisted the obs_key in atari only')

        # push both to hf for seen tasks; push only distances in newdata to hf for unseen tasks
        if task in SEEN_TASK_NAME_TO_ENV_ID:
            subset = Dataset.from_dict(subset_data)
            subset.push_to_hub(repo_id, config_name=f"{task}_subset")
            newdata = Dataset.from_dict(new_data)
            newdata.push_to_hub(repo_id, config_name=f"{task}_newdata")
        elif task in UNSEEN_TASK_NAME_TO_ENV_ID:
            del new_data['indices']
            newdata = Dataset.from_dict(new_data)
            newdata.push_to_hub(repo_id, config_name=f"{task}_newdata")

if __name__ == '__main__':
    main()
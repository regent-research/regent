import torch 
import numpy as np
from datasets import load_from_disk
from datasets.config import HF_DATASETS_CACHE
from regent.utils import myprint, get_task_info, load_all_row_idxs_lookup, get_dist_stats, get_optional_suffix
from regent.atari_utils import convert_local_to_global_action, convert_global_to_local_action
import json 
import os 
from jat.eval.rl import TASK_NAME_TO_ENV_ID
from regent.eval.rl import UNSEEN_TASK_NAME_TO_ENV_ID, SEEN_TASK_NAME_TO_ENV_ID


LOSS_WEIGHTS = {
    **{task: 1.0 for task in TASK_NAME_TO_ENV_ID.keys() if task.startswith("mujoco")},
    **{task: 1.0 for task in TASK_NAME_TO_ENV_ID.keys() if task.startswith("metaworld")},
    **{task: 1.0 for task in TASK_NAME_TO_ENV_ID.keys() if task.startswith("babyai")},
    **{task: 1.0 for task in TASK_NAME_TO_ENV_ID.keys() if task.startswith("atari")},
}


class RetrievalAugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, task, ONLY_RL_TASKS, num_contexts, lamda, use_global_atari_actions, dist_multipliers, dist_normalizer, atari_dist_type, use_atari_embeddings, finetune_num_demos):
        # init
        optional_suffix = get_optional_suffix(task, atari_dist_type, finetune_num_demos)
        folder_loc = f"{HF_DATASETS_CACHE}/regent-research/regent-subset-of-jat-dataset-tokenized-local/{task}{optional_suffix}"
        rew_key, attn_key, obs_key, act_key, B, obs_dim, act_dim = get_task_info(task)
        extra_key = 'discrete_RandP_action_logits' if task.startswith("atari") or task.startswith("babyai") else 'continuous_RandP_actions'
        local_path = f'dataset_jat_regent/{task}'
        all_row_idxs_lookup = load_all_row_idxs_lookup(task, local_path, optional_suffix=optional_suffix)

        # load shapes and bin files
        assert os.path.exists(folder_loc), f'{folder_loc} does not exist'
        with open(f"{folder_loc}/shapes.json", 'r') as f:
            all_shapes = json.load(f)
        all_shapes = {k: tuple(v_list) for k, v_list in all_shapes.items() if k != 'max_dist'}
        if use_atari_embeddings and task.startswith("atari"):
            obs_key = f'embeddings_resnet18_512'
            all_shapes[obs_key] = (all_shapes[act_key][0], all_shapes[act_key][1], 512)
        all_data = {k: np.memmap(f"{folder_loc}/{k}.bin", dtype='int32' if k in ['indices', 'discrete_actions', 'discrete_observations'] else 'float32', mode='r', shape=all_shapes[k])
                    for k in [obs_key, act_key, rew_key, 'distances', 'indices']}
        mean_dist, std_dist, max_dist, p80, p85, p90, p95, p99 = get_dist_stats(task=task, optional_suffix=optional_suffix)
        
        ###### information for the reader of this code on the shapes of data
        num_all_row_idxs = len(all_row_idxs_lookup) # the number of rows in the jat-dataset-tokenized that were used to create this dataset; each row has a maximum of B states.
        num_total = all_shapes['indices'][0] # the total number of states in the above rows; also the length of this dataset
        if task.startswith("atari"): # overwrite obs_dim because raw obs in atari are (4, 84, 84) and raw obs in babyai have 64 extra dim
            if use_atari_embeddings and task.startswith("atari"):
                obs_dim = (512,)
            else:
                obs_dim = (4, 84, 84)
        elif task.startswith("babyai"):
            obs_dim = (obs_dim[0]+64,)
        assert all_shapes[obs_key] == (num_all_row_idxs, B, *obs_dim)
        assert ((act_key == 'continuous_actions' and all_shapes[act_key] == (num_all_row_idxs, B, act_dim)) or 
                (act_key == 'discrete_actions' and all_shapes[act_key] == (num_all_row_idxs, B)))
        assert all_shapes[rew_key] == (num_all_row_idxs, B)
        assert all_shapes['distances'] == (num_total, num_contexts)
        assert all_shapes['indices'] == (num_total, num_contexts, 2)

        # carry over to self
        self.task = task
        self.num_contexts = num_contexts
        self.lamda = lamda
        self.use_global_atari_actions = use_global_atari_actions
        self.dist_multipliers = dist_multipliers
        self.vocab_size = 18 if ONLY_RL_TASKS else 50257
        self.N = self.vocab_size
        self.obs_key = obs_key
        self.act_key = act_key
        self.rew_key = rew_key
        self.attn_key = attn_key
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.lw_key = 'loss_weight'
        self.extra_key = extra_key
        self.all_row_idxs_lookup = all_row_idxs_lookup
        self.all_shapes = all_shapes
        self.num_all_row_idxs = num_all_row_idxs
        self.num_total = num_total
        self.all_data = all_data
        self.max_dist = max_dist
        self.mean_dist = mean_dist
        self.std_dist = std_dist
        self.p80, self.p85, self.p90, self.p95, self.p99 = p80, p85, p90, p95, p99
        self.dist_normalizer_value = {'std': std_dist, 'max': max_dist, 'p80': p80, 'p85': p85, 'p90': p90, 'p95': p95, 'p99': p99}[dist_normalizer]
        if self.dist_normalizer_value == 0.0: self.dist_normalizer_value = 1.0
        self.use_atari_embeddings = use_atari_embeddings

    def __len__(self):
        return self.all_shapes['indices'][0]

    def __getitem__(self, idx):
        # distances and indices for this idx can be directly accessed
        distances = self.all_data['distances'][idx]
        indices = self.all_data['indices'][idx]

        # distances: divide by std
        distances = distances / self.dist_normalizer_value
        if self.task.startswith("mujoco"):
            distances = distances * self.dist_multipliers['mujoco']
        elif self.task.startswith("atari"):
            distances = distances * self.dist_multipliers['atari']

        # setup outputs
        outputs = {}
        for k in [self.obs_key, self.act_key, self.rew_key]:
            outputs[k] = []
        outputs['exp_lamda_distances'] = np.exp(- self.lamda * distances)[:, np.newaxis] # expands to (num_contexts, 1)
        outputs[self.attn_key] = np.array([1 for _ in range(self.num_contexts)]).astype(np.float32)
        outputs[self.lw_key] = np.array([LOSS_WEIGHTS[self.task] for _ in range(self.num_contexts)]).astype(np.float32)

        # get all row and column indices and fill up the output keys with empty lists
        for actual_row_idx, actual_col_idx in indices:
            r = self.all_row_idxs_lookup[actual_row_idx]
            c = actual_col_idx
            outputs[self.obs_key].append(self.all_data[self.obs_key][r][c])
            temp_a = self.all_data[self.act_key][r][c]
            if self.task.startswith("atari") and self.use_global_atari_actions:
                temp_a = convert_local_to_global_action( temp_a, self.task )
            outputs[self.act_key].append(temp_a)
            outputs[self.rew_key].append(self.all_data[self.rew_key][r][c])
        outputs[self.obs_key] = np.stack(outputs[self.obs_key])
        outputs[self.act_key] = np.stack(outputs[self.act_key])
        outputs[self.rew_key] = np.stack(outputs[self.rew_key])

        # fill up the extra_key in outputs
        RandP_action = outputs[self.act_key][0]
        if self.extra_key == 'continuous_RandP_actions':
            outputs[self.extra_key] = [RandP_action for _ in range(self.num_contexts)]
        elif self.extra_key == 'discrete_RandP_action_logits':
            outputs[self.extra_key] = []
            for d in distances:
                d = min(1.0, max(0.0, d))
                curr_logits = [1.0/self.N * d for _ in range(self.N)]
                curr_logits[RandP_action] = (1.0 + (self.N - 1.0)*(1.0 - d))/self.N
                outputs[self.extra_key].append(curr_logits)
        outputs[self.extra_key] = np.stack(outputs[self.extra_key]).astype(np.float32)

        # convert to tensors and print all dtypes
        for k in outputs.keys():
            outputs[k] = torch.from_numpy(outputs[k])
            if k in ['discrete_actions', 'discrete_observations']:
                outputs[k] = outputs[k].long()

        # change keys for embeddings
        if self.use_atari_embeddings and self.task.startswith("atari"):
            outputs['continuous_observations'] = outputs[self.obs_key]
            del outputs[self.obs_key]
        
        return outputs



class CombinedRetrievalAugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, tasks, ONLY_RL_TASKS, n_contiguous, split, num_contexts, lamda, use_global_atari_actions, dist_multipliers, dist_normalizer, atari_dist_type, use_atari_embeddings, finetune_num_demos):
        # iterate over all tasks and save dataset object, combinations of task and indices
        all_datasets = {}
        all_combos = []
        if split == 'train':
            rng = np.random.default_rng(seed=27)
        for task in tasks:
            myprint(f"<Setting up dataset for split: {split}, task: {task}>")                
            if task in ['oscar', 'wikipedia', 'conceptual-captions', 'ok-vqa']:
                myprint(f'WARNING: non RL dataset from {task}')
                dataset = load_from_disk(f"{HF_DATASETS_CACHE}/jat-project/jat-dataset-tokenized/{task}")
                dataset = dataset['train']
            else:
                dataset = RetrievalAugmentedDataset(task=task, ONLY_RL_TASKS=ONLY_RL_TASKS, num_contexts=num_contexts, lamda=lamda, use_global_atari_actions=use_global_atari_actions, dist_multipliers=dist_multipliers, dist_normalizer=dist_normalizer, atari_dist_type=atari_dist_type, use_atari_embeddings=use_atari_embeddings, finetune_num_demos=finetune_num_demos)
            all_datasets[task] = dataset

            # set NUM_EVAL: for finetuning on unseen tasks, we don't need eval split
            NUM_EVAL = 0 if task in UNSEEN_TASK_NAME_TO_ENV_ID else 2000

            # set up start and end; ensuring both are multiples of n_contiguous
            if split == 'train':
                s, e = 0, ((len(dataset) - NUM_EVAL) // n_contiguous) * n_contiguous # e.g: 0, 98k // 1024 * 1024 = 0, 97280
            elif split == 'eval':
                s, e = ((len(dataset) - NUM_EVAL) // n_contiguous) * n_contiguous, (len(dataset) // n_contiguous) * n_contiguous # e.g: 98k // 1024 * 1024, 100k // 1024 * 1024 = 97280, 99328
            else:
                raise ValueError(f"split has to be in ['train', 'eval'], but got {split}")
            
            # split range(s, e) indices into groups of size n_contiguous 
            indices = list(range(s, e))
            if split == 'train': # shuffle the indices; can remove if you want to keep the order in batches during training
                rng.shuffle(indices)
            for i in range(0, len(indices), n_contiguous):
                group = [(task, k) for k in indices[i:i+n_contiguous]]
                all_combos.append(group)

        # if train split, shuffle all_combos
        if split == 'train':
            rng.shuffle(all_combos)

        # flatten all_combos
        all_combos = [item for sublist in all_combos for item in sublist]

        # save
        self.all_datasets = all_datasets
        self.all_combos = all_combos

    def __len__(self):
        return len(self.all_combos)
    
    def __getitem__(self, idx):
        task, this_idx = self.all_combos[idx]
        return self.all_datasets[task][this_idx]
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from torch import BoolTensor, FloatTensor, LongTensor, Tensor, nn
from transformers import GPTNeoModel, GPTNeoPreTrainedModel
from transformers.modeling_outputs import ModelOutput
from transformers.models.vit.modeling_vit import ViTPatchEmbeddings

from jat.configuration_jat import JatConfig
from jat.processing_jat import JatProcessor
from regent.utils import build_index_vector, get_task_info, process_obs_and_collect_indices, collect_all_data, process_row_of_obs_atari_full_without_mask, retrieve_atari, retrieve_vector, myprint, save_images_side_by_side, L2dist, get_images_of_retrieved_obs, get_emb_transform_model_dim
from PIL import Image
import os
from copy import deepcopy


class RandP():
    def __init__(self, 
                 task,
                 dataset, 
                 num_demos, # to retrieve from
                 device,
                 atari_dist_type,
                 batch_size_retrieval=16, # for atari envs on gpu
                 nb_cores_autofaiss=8, # for vector obs envs on cpu cores
        ) -> None:
        # setup
        rew_key, attn_key, obs_key, act_key, B, obs_dim, act_dim = get_task_info(task)
        extra_key = 'discrete_RandP_action_logits' if task.startswith("atari") or task.startswith("babyai") else 'continuous_RandP_actions'

        # get embedding model
        if task.startswith("atari"):
            self.device = device
            self.emb_transform, self.emb_model, emb_dim = get_emb_transform_model_dim(atari_dist_type, self.device)
            obs_dim = emb_dim # overwrite for atari_dist_type

        kwargs = {'B': B,
              'obs_dim': obs_dim,
              'attn_key': attn_key,
              'obs_key': obs_key,
              'device': device,
              'task': task,
              'batch_size_retrieval': batch_size_retrieval,
              'nb_cores_autofaiss': nb_cores_autofaiss,
              'verbose': True,
              'atari_dist_type': atari_dist_type,
            }
        raw_obs_dim = obs_dim
        if task.startswith("atari"): # overwrite raw_obs_dim because raw obs in atari are (4, 84, 84) and raw obs in babyai have 64 extra dim
            raw_obs_dim = (4, 84, 84)
        elif task.startswith("babyai"):
            raw_obs_dim = (obs_dim[0]+64,)
        
        # save
        self.task = task
        self.dataset = dataset
        self.obs_key = obs_key
        self.act_key = act_key
        self.rew_key = rew_key
        self.attn_key = attn_key
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.extra_key = extra_key
        self.kwargs = kwargs
        self.raw_obs_dim = raw_obs_dim
        
        # for retrieval,
        all_rows_of_obs_OG, all_attn_masks_OG, all_row_idxs, all_datarows_dict = collect_all_data(dataset, task, obs_key, num_demos, return_datarows_dict=True, atari_dist_type=atari_dist_type)
        if task.startswith("babyai"):
            # for each mission in task,
            self.all_indices = {}
            self.knn_index = {}
            for mission_idx, mission in enumerate(all_row_idxs.keys()):
                # create index, collect subset of data that we can retrieve from
                myprint(('*'*50) + f'{mission=} - {mission_idx+1}/{len(all_row_idxs.keys())}')
                self.all_indices[mission], self.knn_index[mission] = build_index_vector(all_rows_of_obs_OG=all_rows_of_obs_OG[mission],
                                                                                        all_attn_masks_OG=all_attn_masks_OG[mission],
                                                                                        all_row_idxs=all_row_idxs[mission],
                                                                                        kwargs=kwargs)
        else:
            # create index, collect subset of data that we can retrieve from
            self.all_indices, self.knn_index = build_index_vector(all_rows_of_obs_OG=all_rows_of_obs_OG,
                                                                  all_attn_masks_OG=all_attn_masks_OG,
                                                                  all_row_idxs=all_row_idxs,
                                                                  kwargs=kwargs)
        
        # for retrieval inside retrieve()
        self.datarows = all_datarows_dict

    def reset_rl(self):
        self.steps = 0

    def process(
        self,
        processor: JatProcessor,
        continuous_observation: Optional[List[float]] = None,
        discrete_observation: Optional[List[int]] = None,
        text_observation: Optional[str] = None,
        image_observation: Optional[np.ndarray] = None,
        action_space: Union[spaces.Box, spaces.Discrete] = None,
        reward: Optional[float] = None,
        deterministic: bool = False,
        context_window: Optional[int] = None,
    ):
        # Get the maximum sequence length
        ### see script/train_jat.py > L161. 
        ### None ==> value set to 512 in jat/processing_jat.py > L354 and then // 2 in L355.
        ### weirdly, the value in script/eval_jat.py is set as 256 so it will be // 2 again in L355.
        max_length = 64 if self.task.startswith("atari") else None 
        
        # Convert everything to lists
        def to_list(x):
            return x.tolist() if isinstance(x, np.ndarray) else x

        continuous_observation = to_list(continuous_observation)
        discrete_observation = to_list(discrete_observation)

        # get babyai mission within task
        if self.task.startswith("babyai"):
            mission = deepcopy(text_observation)
            assert mission in self.knn_index.keys(), f'{mission=} should be in {self.knn_index.keys()=}'

        # Add a fake action to the end of the sequence
        if isinstance(action_space, spaces.Box):
            fake_continuous_action = [0.0 for _ in range(action_space.shape[0])]
            fake_discrete_action = None
        elif isinstance(action_space, spaces.Discrete):
            fake_continuous_action = None
            fake_discrete_action = 0

        continuous_observations = [continuous_observation] if continuous_observation is not None else None
        discrete_observations = [discrete_observation] if discrete_observation is not None else None
        text_observations = [text_observation] if text_observation is not None else None
        image_observations = [image_observation] if image_observation is not None else None
        continuous_actions = [fake_continuous_action] if fake_continuous_action is not None else None
        discrete_actions = [fake_discrete_action] if fake_discrete_action is not None else None
        rewards = [reward] if reward is not None else [0.0]

        # Add the batch dimension
        continuous_observations = [continuous_observations] if continuous_observations is not None else None
        discrete_observations = [discrete_observations] if discrete_observations is not None else None
        text_observations = [text_observations] if text_observations is not None else None
        image_observations = [image_observations] if image_observations is not None else None
        continuous_actions = [continuous_actions] if continuous_actions is not None else None
        discrete_actions = [discrete_actions] if discrete_actions is not None else None
        rewards = [rewards]

        # Process the inputs
        processed = processor(
            continuous_observations=continuous_observations,
            discrete_observations=discrete_observations,
            text_observations=text_observations,
            image_observations=image_observations,
            continuous_actions=continuous_actions,
            discrete_actions=discrete_actions,
            rewards=rewards,
            truncation=True,
            truncation_side="left",
            max_length=max_length,
            return_tensors="pt",
        )

        assert (((self.act_key == 'continuous_actions' and processed[self.act_key].shape == (1, 1, self.act_dim)) or 
                 (self.act_key == 'discrete_actions' and processed[self.act_key].shape == (1, 1))) and
                processed[self.obs_key].shape == (1, 1, *self.raw_obs_dim) and
                processed[self.rew_key].shape == (1, 1)), f'{processed[self.act_key].shape=}, {processed[self.obs_key].shape=}, {processed[self.rew_key].shape=}, {self.act_dim=}, {self.raw_obs_dim=}'

        # save babyai mission
        if self.task.startswith("babyai"):
            processed['mission'] = mission

        return processed

    def retrieve(
        self,
        all_processed: List[dict],
        num_to_retrieve: int,
    ):
        self.steps += 1
        # Set num envs
        num_envs = len(all_processed)

        # Get obs from processed and make batch
        row_of_obs = [all_processed[idx][self.obs_key][0].numpy() for idx in range(num_envs)]
        row_of_obs = np.concatenate(row_of_obs)
        assert row_of_obs.shape == (num_envs, *self.raw_obs_dim) and isinstance(row_of_obs, np.ndarray)
        if self.task.startswith("atari"):
            row_of_obs = process_row_of_obs_atari_full_without_mask(row_of_obs)
            row_of_obs = torch.from_numpy(row_of_obs).to(self.device)
            with torch.no_grad():
                row_of_obs = self.emb_model(self.emb_transform(row_of_obs)).cpu().numpy()
        elif self.task.startswith("babyai"):
            row_of_obs = row_of_obs[:, :148] # removing last 64 text tokens
        assert row_of_obs.shape == (num_envs, *self.obs_dim) and isinstance(row_of_obs, np.ndarray)

        # Retrieve indices
        if self.task.startswith("babyai"):
            retrieved_indices = []
            for idx in range(num_envs):
                mission = all_processed[idx]['mission']
                retrieved_indices_mission = retrieve_vector(row_of_obs=row_of_obs[idx:idx+1],
                                                            knn_index=self.knn_index[mission], 
                                                            all_indices=self.all_indices[mission], 
                                                            num_to_retrieve=num_to_retrieve,
                                                            kwargs=self.kwargs)
                retrieved_indices.append(retrieved_indices_mission) # appending (1, 1, 2)
            retrieved_indices = np.concatenate(retrieved_indices, axis=0)
            assert retrieved_indices.shape == (num_envs, num_to_retrieve, 2)
        else:
            retrieved_indices = retrieve_vector(row_of_obs=row_of_obs, 
                                                knn_index=self.knn_index, 
                                                all_indices=self.all_indices, 
                                                num_to_retrieve=num_to_retrieve,
                                                kwargs=self.kwargs)

        # Corresponding row and column indices
        all_row_idx_and_i = retrieved_indices[:, 0, :] # 0 since we only need first retrieved index for RandP
        
        # Return action
        all_retrieved_act = []
        all_retrieved_obs = []
        all_retrieved_rew = []
        env_idx = 0
        for row_idx, i in all_row_idx_and_i:
            if self.task.startswith("babyai"):
                mission = all_processed[env_idx]['mission']
                datarow = self.datarows[mission][int(row_idx)]
            else:
                datarow = self.datarows[int(row_idx)]
            all_retrieved_act.append(datarow[self.act_key][int(i)])
            all_retrieved_obs.append(datarow[self.obs_key][int(i)])
            all_retrieved_rew.append(datarow[self.rew_key][int(i)])
            env_idx += 1

        ### analysis?
        # if self.task.startswith("atari"):
        #     row_of_retrieved_obs = process_row_of_obs_atari_full_without_mask(all_retrieved_obs)
        #     assert row_of_obs.shape == (num_envs, *self.obs_dim) and isinstance(row_of_obs, np.ndarray)
        #     myprint(f'{all_retrieved_act=}')
        #     # save images
        #     os.makedirs(f'outputs/RandP/{self.task}', exist_ok=True)
        #     to_save = []
        #     for idx in range(num_envs):
        #         to_save.append(row_of_obs[idx])
        #         to_save.append(row_of_retrieved_obs[idx])
        #         to_save.append(np.zeros_like(row_of_obs[idx]))
        #     save_images_side_by_side(to_save, save_path=f'outputs/RandP/{self.task}/{self.steps}.png')

        return all_retrieved_act, all_retrieved_obs, all_retrieved_rew
    
    def get_next_action(
        self,
        all_processed: List[dict],
        return_retrieved_obs: bool = False,
    ):
        num_envs = len(all_processed)

        # Get the retrieved data
        all_retrieved_act, all_retrieved_obs, all_retrieved_rew = self.retrieve(all_processed, num_to_retrieve=1)

        if return_retrieved_obs:
            all_retrieved_images = get_images_of_retrieved_obs(deepcopy(all_retrieved_obs), self.task)

        return all_retrieved_act if not return_retrieved_obs else (all_retrieved_act, all_retrieved_images)

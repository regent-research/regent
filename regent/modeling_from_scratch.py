import torch
import numpy as np
from torch import BoolTensor, FloatTensor, LongTensor
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union
from regent.utils import get_task_info
from jat.modeling_jat import JatOutput
from jat.processing_jat import JatProcessor
from gymnasium import spaces
from copy import deepcopy


# IMPALA-CNN from CleanRL: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/qdagger_dqn_atari_impalacnn.py
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs

class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)


class ImpalaCNN(nn.Module):
    def __init__(self, task, shape=(4, 84, 84), num_actions=18):
        super().__init__()
        self.task = task
        rew_key, attn_key, obs_key, act_key, B, obs_dim, act_dim = get_task_info(task)
        self.obs_key = obs_key
        self.num_actions = num_actions
        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=num_actions),
        ]
        self.network = nn.Sequential(*conv_seqs)

    def forward(self, 
                continuous_observations: Optional[FloatTensor] = None,
                discrete_observations: Optional[LongTensor] = None,
                image_observations: Optional[FloatTensor] = None, # [batch_size, 4, 84, 84]
                continuous_actions: Optional[FloatTensor] = None,
                discrete_actions: Optional[LongTensor] = None, # [batch_size,]
                rewards: Optional[FloatTensor] = None,
                attention_mask: Optional[BoolTensor] = None,
                loss_weight: Optional[FloatTensor] = None,):
        
        pred_logits = self.network(image_observations) # [batch_size, num_actions]
        if discrete_actions is not None:
            action_loss = F.cross_entropy(pred_logits, discrete_actions)
        else:
            action_loss = None

        return JatOutput(
            loss=action_loss,
            observation_loss=0.0,
            action_loss=action_loss,
            pred_observations=None,
            pred_actions=pred_logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
    
    @torch.no_grad()
    def get_next_action(self,
                        all_processed: List[dict]):
        num_envs = len(all_processed)
        all_query_obs = torch.stack([all_processed[idx][self.obs_key][0, 0] for idx in range(num_envs)]).to(self.device)
        assert all_query_obs.shape == (num_envs, 4, 84, 84)
        outputs = self.forward(**{self.obs_key: all_query_obs})
        logits = outputs.pred_actions
        assert logits.shape == (num_envs, self.num_actions)
        last_discrete_action = logits.argmax(dim=-1, keepdim=True).cpu().numpy().reshape(-1)
        return last_discrete_action


# Vector Observation Environments
class BC_MLP(nn.Module):
    def __init__(self, task, hidden_dims=[256, 256]):
        super().__init__()
        self.task = task
        rew_key, attn_key, obs_key, act_key, B, obs_dim, act_dim = get_task_info(task)
        obs_dim = obs_dim[0] # taking first item of one-item tuple
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obs_key = obs_key
        layers = [nn.Linear(obs_dim, hidden_dims[0]), nn.ReLU()]
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], act_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, 
                continuous_observations: Optional[FloatTensor] = None, # [batch_size, obs_dim]
                discrete_observations: Optional[LongTensor] = None,
                image_observations: Optional[FloatTensor] = None,
                continuous_actions: Optional[FloatTensor] = None, # [batch_size, act_dim]
                discrete_actions: Optional[LongTensor] = None,
                rewards: Optional[FloatTensor] = None,
                attention_mask: Optional[BoolTensor] = None,
                loss_weight: Optional[FloatTensor] = None,):
        
        pred_actions = self.network(continuous_observations) # [batch_size, act_dim]
        if continuous_actions is not None:
            action_loss = F.mse_loss(pred_actions, continuous_actions)
        else:
            action_loss = None

        return JatOutput(
            loss=action_loss,
            observation_loss=0.0,
            action_loss=action_loss,
            pred_observations=None,
            pred_actions=pred_actions,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
    
    @torch.no_grad()
    def get_next_action(self,
                        all_processed: List[dict]):
        num_envs = len(all_processed)
        all_query_obs = torch.stack([all_processed[idx][self.obs_key][0, 0] for idx in range(num_envs)]).to(self.device)
        assert all_query_obs.shape == (num_envs, self.obs_dim)
        outputs = self.forward(**{self.obs_key: all_query_obs})
        preds = outputs.pred_actions
        assert preds.shape == (num_envs, self.act_dim)
        return list(preds.cpu().numpy())


# Common process function
def process(
    task: str,
    processor: JatProcessor,
    continuous_observation: Optional[List[float]] = None,
    discrete_observation: Optional[List[int]] = None,
    text_observation: Optional[str] = None,
    image_observation: Optional[np.ndarray] = None,
    action_space: Union[spaces.Box, spaces.Discrete] = None,
    reward: Optional[float] = None,
    deterministic: bool = True,
    context_window: Optional[int] = None,
):
    # Get task info
    rew_key, attn_key, obs_key, act_key, B, obs_dim, act_dim = get_task_info(task)
    raw_obs_dim = obs_dim
    if task.startswith("atari"): # overwrite raw_obs_dim because raw obs in atari are (4, 84, 84) and raw obs in babyai have 64 extra dim
        raw_obs_dim = (4, 84, 84)
    elif task.startswith("babyai"):
        raw_obs_dim = (obs_dim[0]+64,)

    # Get the maximum sequence length
    max_length = context_window
    
    # Convert everything to lists
    def to_list(x):
        return x.tolist() if isinstance(x, np.ndarray) else x

    continuous_observation = to_list(continuous_observation)
    discrete_observation = to_list(discrete_observation)

    # get babyai mission within task
    if task.startswith("babyai"):
        mission = deepcopy(text_observation)

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

    assert (((act_key == 'continuous_actions' and processed[act_key].shape == (1, 1, act_dim)) or # zeros
                (act_key == 'discrete_actions' and processed[act_key].shape == (1, 1))) and
            processed[obs_key].shape == (1, 1, *raw_obs_dim) and
            processed[rew_key].shape == (1, 1)), f'{processed[act_key].shape=}, {processed[obs_key].shape=}, {processed[rew_key].shape=}, {act_dim=}, {raw_obs_dim=}'

    # save babyai mission
    if task.startswith("babyai"):
        processed['mission'] = mission

    # save action_space and deterministic
    processed['action_space'] = action_space
    processed['deterministic'] = deterministic

    return processed
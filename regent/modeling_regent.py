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
import torch.nn.functional as F


from jat.configuration_jat import JatConfig
from jat.processing_jat import JatProcessor
from jat.modeling_jat import JatModel, compute_mse_loss, cyclic_expand_dim, JatOutput
from regent.utils import build_index_vector, get_task_info, collect_all_data, process_row_of_obs_atari_full_without_mask, retrieve_vector, myprint, L2dist, get_dist_stats, get_images_of_retrieved_obs, get_emb_transform_model_dim, get_optional_suffix
from regent.atari_utils import convert_local_to_global_action, convert_global_to_local_action
from regent.eval.rl import SEEN_TASK_NAME_TO_ENV_ID, UNSEEN_TASK_NAME_TO_ENV_ID
from PIL import Image
import os
from copy import deepcopy
from pytorch_msssim import ssim
import json


def cross_entropy_from_softmax(softmax_probs, targets, reduction="mean", epsilon=1e-9):
    """
    Calculate the cross entropy loss given softmax_probs and targets.

    :param softmax_probs: tensor containing softmax probabilities
    :param targets: tensor containing the target classes (not one-hot encoded)
    :return: cross entropy loss
    """
    assert len(softmax_probs.shape) == 2, "softmax_probs should be of shape (batch_size, num_classes)"
    assert len(targets.shape) == 1, "targets should be of shape (batch_size,)"

    # Convert targets to one-hot encoding
    targets_one_hot = F.one_hot(targets, num_classes=softmax_probs.shape[1]).float() # shape: (batch_size, num_classes)
    
    # Calculate the cross entropy loss
    softmax_probs = softmax_probs.clamp(min=epsilon, max=1-epsilon) # to avoid NaNs from log(0) and instabilities from log(1)
    log_softmax_probs = softmax_probs.log()  # safe to take log as softmax_probs are non-zero
    loss = -torch.sum(targets_one_hot * log_softmax_probs, dim=1)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError("reduction should be one of 'mean', 'sum', or 'none'")


def compute_ce_loss_from_softmax(
    logits: FloatTensor, labels: torch.LongTensor, mask: Optional[BoolTensor], weights: Optional[FloatTensor] = None
) -> FloatTensor:
    """
    Compute the Cross Entropy (CE) loss between predicted logits and true class labels, considering valid timesteps.

    Args:
        logits (`FloatTensor` of shape `(batch_size, max_seq_len, [inner_size,] num_classes)`):
            Predicted logits at the output of the model.
        labels (`torch.LongTensor` of shape `(batch_size, max_seq_len, [inner_size,])`):
            Ground truth class labels.
        mask (`BoolTensor` of shape `(batch_size, max_seq_len)`, *optional*):
            Boolean mask indicating valid timesteps.
        weights (`FloatTensor` of shape `(batch_size, max_seq_len)`, *optional*):
            Weights to be applied to the loss.

    Returns:
        loss (`FloatTensor` of shape `(,)`):
            CE loss between predicted logits and true class labels.
    """
    if mask is not None:
        logits = logits[mask.bool()]  # (Y, X, C)
        labels = labels[mask.bool()]  # (Y, X)
        if weights is not None:
            weights = weights[mask.bool()]  # (Y,)
    else:
        logits = logits.flatten(end_dim=2)  # (B, L, X, C) -> (B*L, X, C)
        labels = labels.flatten(end_dim=1)  # (B, L, X) -> (B*L, X)
        if weights is not None:
            weights = weights.flatten(end_dim=1)  # (B, L) -> (B*L,)

    loss = cross_entropy_from_softmax(logits.view(-1, logits.size(-1)), labels.view(-1), reduction="none")  # (Y*X,) # we don't use F.cross_entropy here to avoid double softmax
    loss = loss.view(labels.size())  # (Y, X)
    loss = loss.mean(-1)  # (Y,)

    # Multiply the loss by the weights
    if weights is not None:
        loss = loss * weights  # (Y,)

    # Average the loss
    loss = loss.mean()

    return loss


def crazy_relu(x, beta):
    return nn.LeakyReLU(beta)(x) - (1-beta) * nn.ReLU()(x-1)


class JatRegentModel(JatModel):
    """
    Jat Regent model.
    """
    def __init__(self, config: JatConfig) -> None:
        super().__init__(config)
        hidden_size = config.hidden_size
        action_vocab_size = config.action_vocab_size

        if config.ONLY_RL_TASKS:
            self.single_discrete_decoder = nn.Linear(hidden_size, action_vocab_size, bias=False)
            self.N = config.action_vocab_size
        else:
            self.N = config.vocab_size
        self.multi_discrete_decoder = None # not needed
        self.image_decoder = None # not needed
        self.num_contexts = config.num_contexts # used in get_next_action() at evaluation in an env only
        self.lamda = config.lamda # used in get_next_action() at evaluation in an env only
        self.use_global_atari_actions = config.use_global_atari_actions
        self.dist_multipliers = {'mujoco': config.mujoco_dist_multiplier, 'atari': config.atari_dist_multiplier}
        self.dist_normalizer = config.dist_normalizer
        self.atari_dist_type = config.atari_dist_type
        self.use_atari_embeddings = config.use_atari_embeddings
        self.finetune_num_demos = config.finetune_num_demos if hasattr(config, 'finetune_num_demos') else None
        if self.use_atari_embeddings:
            self.image_encoder = None
            self.emb_dim_full = (512,)

        # print number of parameters
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        myprint(f"number of parameters: {num_params / 1e6:.4f}M")

    def retrieval_setup(self,
                        task,
                        dataset, 
                        num_demos, # to retrieve from
                        device,
                        batch_size_retrieval=16, # for atari envs on gpu
                        nb_cores_autofaiss=8, # for vector obs envs on cpu cores
    ):
        # setup
        rew_key, attn_key, obs_key, act_key, B, obs_dim, act_dim = get_task_info(task)
        extra_key = 'discrete_RandP_action_logits' if task.startswith("atari") or task.startswith("babyai") else 'continuous_RandP_actions'
        optional_suffix = get_optional_suffix(task, self.atari_dist_type, self.finetune_num_demos)
        mean_dist, std_dist, max_dist, p80, p85, p90, p95, p99 = get_dist_stats(task=task, optional_suffix=optional_suffix)
        
        # get embedding model
        if task.startswith("atari"):
            self.emb_transform, self.emb_model, emb_dim, self.emb_model_full = get_emb_transform_model_dim(self.atari_dist_type, self.device, return_emb_weights=True)
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
              'atari_dist_type': self.atari_dist_type,
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
        self.max_dist = max_dist
        self.mean_dist = mean_dist
        self.std_dist = std_dist
        self.p80, self.p85, self.p90, self.p95, self.p99 = p80, p85, p90, p95, p99
        self.dist_normalizer_value = {'std': std_dist, 'max': max_dist, 'p80': p80, 'p85': p85, 'p90': p90, 'p95': p95, 'p99': p99}[self.dist_normalizer]
        if self.dist_normalizer_value == 0.0: self.dist_normalizer_value = 1.0
        
        # for retrieval,
        all_rows_of_obs_OG, all_attn_masks_OG, all_row_idxs, all_datarows_dict = collect_all_data(dataset, task, obs_key, num_demos, return_datarows_dict=True, atari_dist_type=self.atari_dist_type)
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
            

        # # for checking if first env state is similar to retrieval episode's first states
        # if task.startswith("mujoco"):
        #     local_path = f"dataset_jat_regent/{task}"
        #     with open(f"{local_path}/eps_2_rows_tokenized.json", 'r') as f:
        #         eps_2_rows_tokenized = json.load(f)
        #     eps_2_rows_tokenized = {int(k): v for k, v in eps_2_rows_tokenized.items()}
        #     row_idxs_of_first_state_of_demos = [eps_2_rows_tokenized[eps][0] for eps in range(num_demos)]
        #     self.first_states_of_demos = [np.array(dataset['train'][row_idx][obs_key][0]) for row_idx in row_idxs_of_first_state_of_demos]
        # else:
        #     self.first_states_of_demos = None

    def output_rl(
        self,
        transformer_outputs,
        continuous_observations: Optional[FloatTensor] = None,
        discrete_observations: Optional[LongTensor] = None,
        image_observations: Optional[FloatTensor] = None,
        continuous_actions: Optional[FloatTensor] = None,
        discrete_actions: Optional[LongTensor] = None,
        rewards: Optional[FloatTensor] = None,
        attention_mask: Optional[BoolTensor] = None,
        return_loss: bool = True,
        return_dict: Optional[bool] = None,
        loss_weight: Optional[FloatTensor] = None,
        exp_lamda_distances: Optional[FloatTensor] = None,
        continuous_RandP_actions: Optional[FloatTensor] = None,
        discrete_RandP_action_logits: Optional[FloatTensor] = None,
    ):
        hidden_states = transformer_outputs.last_hidden_state
        loss, observation_loss, action_loss = None, None, None
        
        # Observations
        assert rewards is not None
        observations_mask = attention_mask[:, 1::2] if attention_mask is not None else None
        assert self.observation_loss_coef == 0.0, f'{self.observation_loss_coef=} should be 0.0 as we are not predicting observations!'
        # warnings.warn("observation_loss_coef is 0.0, skipping memory-intensive observations prediction.")
        pred_observations = None
        observation_loss = 0.0

        # Actions
        actions_mask = attention_mask[:, ::2] if attention_mask is not None else None
        if continuous_actions is not None:
            act_size = continuous_actions.shape[-1]
            continuous_actions = cyclic_expand_dim(continuous_actions, self.config.max_continuous_size)
            continuous_RandP_actions = cyclic_expand_dim(continuous_RandP_actions, self.config.max_continuous_size)
            init_pred_actions = self.continuous_decoder(hidden_states[:, ::2])
            pred_actions = self.continuous_action_interpolation(init_pred_actions, exp_lamda_distances, continuous_RandP_actions, beta=0.0)
            if return_loss:
                action_loss = compute_mse_loss(pred_actions, continuous_actions, actions_mask, weights=loss_weight) # loss_weight is usually 50 for metaworld, 10 for mujoco (except two tasks where it is 20, 50), 1 for the rest!
            pred_actions = pred_actions[..., :act_size]
        elif discrete_actions is not None:
            init_pred_actions = self.single_discrete_decoder(hidden_states[:, ::2])
            pred_actions = self.discrete_action_interpolation(init_pred_actions, exp_lamda_distances, discrete_RandP_action_logits, beta=0.0)
            if return_loss:
                action_loss = compute_ce_loss_from_softmax(pred_actions, discrete_actions, actions_mask, weights=loss_weight)

        # Return output
        if return_loss:
            loss = self.observation_loss_coef * observation_loss + self.action_loss_coef * action_loss

        if not return_dict:
            output = (pred_observations, pred_actions) + transformer_outputs[1:]
            return ((loss, observation_loss, action_loss) + output) if loss is not None else output

        return JatOutput(
            loss=loss,
            observation_loss=observation_loss,
            action_loss=action_loss,
            pred_observations=pred_observations,
            pred_actions=pred_actions,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
    
    def shifted_crazy_relu(self, x, beta):
        return 2 * crazy_relu(0.5*(x+1), beta) - 1

    def continuous_action_interpolation(self, init_pred_actions, exp_lamda_distances, continuous_RandP_actions, beta=0.0):
        batch_size, max_seq_len, act_size = init_pred_actions.shape
        assert (init_pred_actions.shape == (batch_size, max_seq_len, act_size) and 
                exp_lamda_distances.shape == (batch_size, max_seq_len, 1) and 
                continuous_RandP_actions.shape == (batch_size, max_seq_len, act_size)), f'{init_pred_actions.shape=}, {exp_lamda_distances.shape=}, {continuous_RandP_actions.shape=}, {(batch_size, max_seq_len, act_size)=}'
                
        """ MCNN interpolation (https://arxiv.org/abs/2310.06171) """
        act_fn = self.shifted_crazy_relu
        final_actions = exp_lamda_distances * continuous_RandP_actions + 10.0 * (1 - exp_lamda_distances) * act_fn(init_pred_actions, beta=beta)
        return final_actions
    
    def discrete_action_interpolation(self, init_pred_actions, exp_lamda_distances, discrete_RandP_action_logits, beta=0.0):
        batch_size, max_seq_len, action_vocab_size = init_pred_actions.shape
        assert (init_pred_actions.shape == (batch_size, max_seq_len, action_vocab_size) and 
                exp_lamda_distances.shape == (batch_size, max_seq_len, 1) and 
                discrete_RandP_action_logits.shape == (batch_size, max_seq_len, action_vocab_size)), f'{init_pred_actions.shape=}, {exp_lamda_distances.shape=}, {discrete_RandP_action_logits.shape=}, {(batch_size, max_seq_len, action_vocab_size)=}'
        
        """ MCNN-like interpolation """
        # print(f'{torch.round(discrete_RandP_action_logits[:, -1],decimals=2)=}')
        # print(f'{torch.round(F.softmax(init_pred_actions, dim=-1)[:, -1],decimals=2)=}')
        # print(f'{torch.round(exp_lamda_distances[:, -1],decimals=2)=}')
        # print(f'first term: {torch.round((exp_lamda_distances * discrete_RandP_action_logits)[:, -1],decimals=2)}')
        # print(f'second term: {torch.round(((1 - exp_lamda_distances) * F.softmax(init_pred_actions, dim=-1))[:, -1],decimals=2)}')
        final_actions = exp_lamda_distances * discrete_RandP_action_logits + (1 - exp_lamda_distances) * F.softmax(init_pred_actions, dim=-1)
        return final_actions
    
    # Copied the forward function from the Parent class with the addition of the last 3 args in the input args and in output_rl args
    def forward(
        self,
        input_ids: Optional[LongTensor] = None,
        pixel_values: Optional[FloatTensor] = None,
        continuous_observations: Optional[FloatTensor] = None,
        discrete_observations: Optional[LongTensor] = None,
        image_observations: Optional[FloatTensor] = None,
        continuous_actions: Optional[FloatTensor] = None,
        discrete_actions: Optional[LongTensor] = None,
        rewards: Optional[FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[FloatTensor]]] = None,
        attention_mask: Optional[BoolTensor] = None,
        token_type_ids: Optional[LongTensor] = None,
        position_ids: Optional[LongTensor] = None,
        return_loss: bool = True,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        loss_weight: Optional[FloatTensor] = None,
        exp_lamda_distances: Optional[FloatTensor] = None,
        continuous_RandP_actions: Optional[FloatTensor] = None,
        discrete_RandP_action_logits: Optional[FloatTensor] = None,
    ) -> JatOutput:
    
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Textual tasks
        if input_ids is not None or pixel_values is not None:
            inputs_embeds, attention_mask = self.embed_textual(input_ids, pixel_values, attention_mask)
        # RL tasks
        elif (
            continuous_observations is not None or discrete_observations is not None or image_observations is not None
        ):
            inputs_embeds, attention_mask = self.embed_rl(
                continuous_observations,
                discrete_observations,
                image_observations,
                continuous_actions,
                discrete_actions,
                rewards,
                attention_mask,
            )
        else:
            raise ValueError("Input not provided.")

        # Pass through transformer
        transformer_outputs = self.transformer(
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if input_ids is not None or pixel_values is not None:
            return self.output_textual(transformer_outputs, input_ids, attention_mask, return_loss, return_dict)
        else:
            return self.output_rl(
                transformer_outputs,
                continuous_observations,
                discrete_observations,
                image_observations,
                continuous_actions,
                discrete_actions,
                rewards,
                attention_mask,
                return_loss,
                return_dict,
                loss_weight,
                exp_lamda_distances,
                continuous_RandP_actions,
                discrete_RandP_action_logits,
            )


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
        deterministic: bool = True,
        context_window: Optional[int] = None,
    ):
        # Get the maximum sequence length
        max_length = self.config.max_position_embeddings // 2

        # Get the maximum sequence length
        ### see script/train_jat.py > L161. 
        ### None ==> value set to 512 in jat/processing_jat.py > L354 and then // 2 in L355.
        ### weirdly, the value in script/eval_jat.py is set as 256 so it will be // 2 again in L355.
        # max_length = 64 if self.task.startswith("atari") else None 
        
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

        assert (((self.act_key == 'continuous_actions' and processed[self.act_key].shape == (1, 1, self.act_dim)) or # zeros
                 (self.act_key == 'discrete_actions' and processed[self.act_key].shape == (1, 1))) and
                processed[self.obs_key].shape == (1, 1, *self.raw_obs_dim) and
                processed[self.rew_key].shape == (1, 1)), f'{processed[self.act_key].shape=}, {processed[self.obs_key].shape=}, {processed[self.rew_key].shape=}, {self.act_dim=}, {self.raw_obs_dim=}'

        # save babyai mission
        if self.task.startswith("babyai"):
            processed['mission'] = mission

        # save action_space and deterministic
        processed['action_space'] = action_space
        processed['deterministic'] = deterministic

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

        # Return action
        all_retrieved_act = []
        all_retrieved_obs = []
        all_retrieved_rew = []
        env_idx = 0
        for all_row_idx_and_i in retrieved_indices:
            all_retrieved_act.append([])
            all_retrieved_obs.append([])
            all_retrieved_rew.append([])
            for row_idx, i in all_row_idx_and_i:
                if self.task.startswith("babyai"):
                    mission = all_processed[env_idx]['mission']
                    datarow = self.datarows[mission][int(row_idx)]
                else:
                    datarow = self.datarows[int(row_idx)]
                temp_a = datarow[self.act_key][int(i)]
                if self.task.startswith("atari") and self.use_global_atari_actions:
                    temp_a = convert_local_to_global_action( temp_a, self.task )
                all_retrieved_act[-1].append(temp_a)
                all_retrieved_obs[-1].append(datarow[self.obs_key][int(i)])
                all_retrieved_rew[-1].append(datarow[self.rew_key][int(i)])
            env_idx += 1

        return all_retrieved_act, all_retrieved_obs, all_retrieved_rew, row_of_obs
    
    def get_distances(
        self,
        all_retrieved_obs: np.ndarray,
        all_processed: List[dict],
        query_obs: np.ndarray,
    ):
        num_envs = len(all_processed)

        # Process retrieved obs like in retrieve
        num_contexts = all_retrieved_obs.shape[1] + 1
        assert all_retrieved_obs.shape == (num_envs, num_contexts - 1, *self.raw_obs_dim) and isinstance(all_retrieved_obs, np.ndarray)
        if self.task.startswith("atari"):
            all_retrieved_obs = all_retrieved_obs.reshape(num_envs * (num_contexts - 1), *self.raw_obs_dim)
            all_retrieved_obs = process_row_of_obs_atari_full_without_mask(all_retrieved_obs)
            all_retrieved_obs = torch.from_numpy(all_retrieved_obs).to(self.device)
            with torch.no_grad():
                all_retrieved_obs = self.emb_model(self.emb_transform(all_retrieved_obs)).cpu().numpy()
            all_retrieved_obs = all_retrieved_obs.reshape(num_envs, num_contexts - 1, *self.obs_dim)
        elif self.task.startswith("babyai"):
            all_retrieved_obs = all_retrieved_obs[:, :, :148]
        assert all_retrieved_obs.shape == (num_envs, num_contexts - 1, *self.obs_dim) and isinstance(all_retrieved_obs, np.ndarray)

        # Compute distances
        all_distances = []
        for idx in range(num_envs):
            first_state = all_retrieved_obs[idx, 0:1]
            distances = [0.0]
            for i in range(1, num_contexts - 1):
                curr_state = all_retrieved_obs[idx, i:i+1]
                dist = L2dist(first_state, curr_state)
                distances.append(dist)
            curr_state = query_obs[idx:idx+1]
            dist = L2dist(first_state, curr_state)
            distances.append(dist)
            all_distances.append(distances)
        all_distances = np.array(all_distances)
        assert all_distances.shape == (num_envs, num_contexts), f'{all_distances.shape=}, {num_envs=}, {num_contexts=}'

        # distances: divide by std
        all_distances = all_distances / self.dist_normalizer_value
        if self.task.startswith("mujoco"):
            all_distances = all_distances * self.dist_multipliers['mujoco']
        elif self.task.startswith("atari"):
            all_distances = all_distances * self.dist_multipliers['atari']
        print(f'{self.dist_normalizer_value=}')
        print(f'{all_distances=}')
        
        return all_distances
    
    @torch.no_grad()
    def get_next_action(
        self,
        all_processed: List[dict],
        return_retrieved_obs: bool = False,
    ):
        num_envs = len(all_processed)
        num_contexts = self.num_contexts

        # Get the retrieved data
        all_retrieved_act, all_retrieved_obs, all_retrieved_rew, row_of_obs = self.retrieve(all_processed, num_to_retrieve=num_contexts - 1)
        if return_retrieved_obs:
            all_retrieved_images = get_images_of_retrieved_obs(deepcopy(all_retrieved_obs), self.task)

        # Get the distances
        all_retrieved_obs = np.stack(all_retrieved_obs).astype(np.int32 if self.obs_key == 'discrete_observations' else np.float32)
        assert all_retrieved_obs.shape == (num_envs, num_contexts - 1, *self.raw_obs_dim), f'{all_retrieved_obs.shape=}, {num_envs=}, {self.raw_obs_dim=}, {num_contexts-1=}'
        all_distances = self.get_distances(all_retrieved_obs=all_retrieved_obs, all_processed=all_processed, query_obs=row_of_obs)

        # Batch retrieved data
        all_retrieved_act = np.stack(all_retrieved_act).astype(np.int32 if self.act_key == 'discrete_actions' else np.float32)
        all_retrieved_rew = np.stack(all_retrieved_rew).astype(np.float32)
        assert (((self.act_key == 'continuous_actions' and all_retrieved_act.shape == (num_envs, num_contexts - 1, self.act_dim)) or 
                 (self.act_key == 'discrete_actions' and all_retrieved_act.shape == (num_envs, num_contexts - 1))) and
                all_retrieved_rew.shape == (num_envs, num_contexts - 1)), f'{all_retrieved_act.shape=}, {all_retrieved_rew.shape=}, {num_envs=}, {self.act_dim=}, {self.raw_obs_dim=}, {num_contexts-1=}'

        # Batch query data (already tensors) # query data is already int32/float32 after processing
        all_query_act = torch.stack([all_processed[idx][self.act_key][0] for idx in range(num_envs)])
        all_query_obs = np.stack([all_processed[idx][self.obs_key][0] for idx in range(num_envs)])
        all_query_rew = torch.stack([all_processed[idx][self.rew_key][0] for idx in range(num_envs)])
        assert (((self.act_key == 'continuous_actions' and all_query_act.shape == (num_envs, 1, self.act_dim)) or 
                 (self.act_key == 'discrete_actions' and all_query_act.shape == (num_envs, 1))) and
                all_query_obs.shape == (num_envs, 1, *self.raw_obs_dim) and
                all_query_rew.shape == (num_envs, 1)), f'{all_query_act.shape=}, {all_query_obs.shape=}, {all_query_rew.shape=}, {num_envs=}, {self.act_dim=}, {self.raw_obs_dim=}'

        # Collect attn
        attn_weights = np.ones((num_envs, num_contexts)).astype(np.float32)
        
        # Compute exp_lamda_distances
        exp_lamda_distances = np.exp(-self.lamda * all_distances)[:, :, np.newaxis]
        assert exp_lamda_distances.shape == (num_envs, num_contexts, 1), f'{exp_lamda_distances.shape=}, {num_envs=}, {num_contexts=}'

        # Compute extra_key
        all_extra_key = []
        for idx in range(num_envs):
            RandP_action = all_retrieved_act[idx, 0]
            if self.extra_key == 'continuous_RandP_actions':
                extra_key = [RandP_action for _ in range(num_contexts)]
            elif self.extra_key == 'discrete_RandP_action_logits':
                extra_key = []
                for d in all_distances[idx]:
                    d = min(1.0, max(0.0, d))
                    curr_logits = [1.0/self.N * d for _ in range(self.N)]
                    curr_logits[RandP_action] = (1.0 + (self.N - 1.0)*(1.0 - d))/self.N
                    extra_key.append(curr_logits)
            extra_key = np.stack(extra_key)
            all_extra_key.append(extra_key)
        all_extra_key = np.stack(all_extra_key).astype(np.float32)
        
        if self.extra_key == 'continuous_RandP_actions':
            assert all_extra_key.shape == (num_envs, num_contexts, self.act_dim), f'{all_extra_key.shape=}, {num_envs=}, {num_contexts=}, {self.act_dim=}'
        elif self.extra_key == 'discrete_RandP_action_logits':
            assert all_extra_key.shape == (num_envs, num_contexts, self.N), f'{all_extra_key.shape=}, {num_envs=}, {num_contexts=}, {self.N=}'

        # Tensorify
        all_retrieved_act = torch.from_numpy(all_retrieved_act)
        all_retrieved_rew = torch.from_numpy(all_retrieved_rew)
        attn_weights = torch.from_numpy(attn_weights).to(self.device)
        exp_lamda_distances = torch.from_numpy(exp_lamda_distances).to(self.device)
        all_extra_key = torch.from_numpy(all_extra_key).to(self.device)

        # Concat retrieved and query batches
        all_act = torch.cat([all_retrieved_act, all_query_act], dim=1).to(self.device)
        all_obs = np.concatenate([all_retrieved_obs, all_query_obs], axis=1)
        if self.use_atari_embeddings and self.task.startswith("atari"):
            all_obs = all_obs.reshape(num_envs * num_contexts, *self.raw_obs_dim)
            all_obs = process_row_of_obs_atari_full_without_mask(all_obs)
            all_obs = torch.from_numpy(all_obs).to(self.device)
            with torch.no_grad():
                all_obs = self.emb_model_full(self.emb_transform(all_obs)).reshape(num_envs, num_contexts, *self.emb_dim_full)
        else:
            all_obs = torch.from_numpy(all_obs).to(self.device)
        all_rew = torch.cat([all_retrieved_rew, all_query_rew], dim=1).to(self.device)
        
        # Collect action_space, deterministic from all_processed
        all_action_space = [all_processed[idx]['action_space'] for idx in range(num_envs)]
        all_deterministic = [all_processed[idx]['deterministic'] for idx in range(num_envs)]
        ## assert that all action_space and deterministic are same for all envs
        assert all([action_space == all_action_space[0] for action_space in all_action_space]), f'{all_action_space=}'
        assert all([deterministic == all_deterministic[0] for deterministic in all_deterministic]), f'{all_deterministic=}'
        ## then just use first one!
        action_space = all_action_space[0]
        deterministic = all_deterministic[0]        

        # Forward pass
        if self.use_atari_embeddings and self.task.startswith("atari"):
            final_obs_key = 'continuous_observations'
        else:
            final_obs_key = self.obs_key
        outputs = self.forward(**{final_obs_key: all_obs, 
                                  self.act_key: all_act, 
                                  self.rew_key: all_rew,
                                  self.attn_key: attn_weights,
                                  'exp_lamda_distances': exp_lamda_distances,
                                  self.extra_key: all_extra_key,
                                }, return_loss=False)

        # Return the predicted action
        if self.act_key == 'continuous_actions':
            self.last_continuous_action = outputs.pred_actions[:, -1].cpu().numpy()
            
            assert self.last_continuous_action.shape == (num_envs, self.act_dim), f'{self.last_continuous_action.shape=}, {num_envs=}, {self.act_dim=}'
            
            myprint(f'L2dist(RandP action, Pred action): {[L2dist(all_retrieved_act[idx, 0].cpu().numpy(), self.last_continuous_action[idx]) for idx in range(num_envs)]}')
            self.last_continuous_action = list(self.last_continuous_action) # list of arrays
            return self.last_continuous_action if not return_retrieved_obs else (self.last_continuous_action, all_retrieved_images)

        elif self.act_key == 'discrete_actions':
            act_n = self.config.action_vocab_size if (self.task.startswith('atari') and self.use_global_atari_actions) else action_space.n
            logits = outputs.pred_actions[:, -1, : act_n]
            assert logits.shape == (num_envs, act_n), f'{logits.shape=}, {num_envs=}, {act_n=}'
            if deterministic:
                # myprint(f'{all_extra_key[:, -1, : action_space.n]=}')
                # myprint(f'{logits=}')
                self.last_discrete_action = logits.argmax(dim=-1, keepdim=True).cpu().numpy().reshape(-1)
            else:  # sample
                self.last_discrete_action = torch.multinomial(logits.softmax(dim=-1), num_samples=1).cpu().numpy().reshape(-1)
            
            assert self.last_discrete_action.shape == (num_envs,), f'{self.last_discrete_action.shape=}, {num_envs=}'

            self.last_discrete_action = list(self.last_discrete_action) # list of ints
            myprint(f'RandP action: {all_retrieved_act[:, 0].cpu().numpy().tolist()} vs Pred action: {self.last_discrete_action}')

            if self.task.startswith("atari") and self.use_global_atari_actions:
                self.last_discrete_action = [convert_global_to_local_action(a, self.task) for a in self.last_discrete_action]
                myprint(f'[IN LOCAL ACTION] RandP action: {[convert_global_to_local_action(a, self.task) for a in all_retrieved_act[:, 0].cpu().numpy().tolist()]} vs Pred action: {self.last_discrete_action}')
                myprint(f'[IN LOCAL ACTION] diff: {[convert_global_to_local_action(a, self.task) - b for a, b in zip(all_retrieved_act[:, 0].cpu().numpy().tolist(), self.last_discrete_action)]}')

            return self.last_discrete_action if not return_retrieved_obs else (self.last_discrete_action, all_retrieved_images)


JatRegentModel.register_for_auto_class("AutoModelForCausalLM")

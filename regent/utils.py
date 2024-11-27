from datetime import datetime
from PIL import PngImagePlugin
import torch
import torch.nn.functional as F
import numpy as np
from pytorch_msssim import ssim
from autofaiss import build_index
from PIL import Image
import os
import time
import json
from typing import List
from datasets.config import HF_DATASETS_CACHE
from regent.eval.rl import SEEN_TASK_NAME_TO_ENV_ID, UNSEEN_TASK_NAME_TO_ENV_ID, make_seen_and_unseen
import psutil
import matplotlib.pyplot as plt
import cv2
from regent.atari_utils import convert_local_to_global_action


def myprint(str):
    # check if first characters of string are newline character
    num_newlines = 0
    while str[num_newlines] == '\n':
        print()
        num_newlines += 1
    str_without_newline = str[num_newlines:]
    print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}:     {str_without_newline}')

def print_memory_usage():
    memory = psutil.virtual_memory()
    print(f"Total memory: {memory.total / (1024**3):.2f} GB")
    print(f"Available memory: {memory.available / (1024**3):.2f} GB")
    print(f"Used memory: {memory.used / (1024**3):.2f} GB")
    print(f"Memory usage: {memory.percent}%")

def log_memory_usage(device, prefix=''):
    allocated = torch.cuda.memory_allocated(device) / 1e9
    max_allocated = torch.cuda.max_memory_allocated(device) / 1e9
    reserved = torch.cuda.memory_reserved(device) / 1e9
    max_reserved = torch.cuda.max_memory_reserved(device) / 1e9
    print(f"{prefix} - Memory - Allocated: {allocated:.2f} GB, Max Allocated: {max_allocated:.2f} GB")
    print(f"{prefix} - Memory - Reserved: {reserved:.2f} GB, Max Reserved: {max_reserved:.2f} GB")

def check_files_with_prefix_suffix(dir, prefix, suffix):
    files = os.listdir(dir)
    matching_files = [f for f in files if f.startswith(prefix) and f.endswith(suffix)]
    if len(matching_files) > 0:
        assert len(matching_files) == 1, f"Multiple files found with in {dir=} with {prefix=}, {suffix=}. These are {matching_files}."
        return True, matching_files[0]
    else:
        return False, None

def get_num_demos(task, finetune_num_demos):
    max_num_demos = 20 if task.startswith("babyai") else 5 if task.startswith("atari") else 100 if task.startswith("metaworld") else 100
    return finetune_num_demos if (finetune_num_demos is not None and task in UNSEEN_TASK_NAME_TO_ENV_ID) else max_num_demos

def get_optional_suffix(task, atari_dist_type, finetune_num_demos):
    num_demos = get_num_demos(task, finetune_num_demos)
    return f'-emb_{atari_dist_type}-{num_demos}demos' if task.startswith('atari') else '' if task.startswith('babyai') else f'-{num_demos}demos'
    
def load_retrieved_indices(task, local_path, atari_dist_type, finetune_num_demos):
    # main code
    num_demos = get_num_demos(task, finetune_num_demos)
    filename = f'retrieved_indices_{num_demos}demos{f"_emb_{atari_dist_type}" if task.startswith("atari") else ""}.json'
    with open(f"{local_path}/{filename}", 'r') as f:
        retrieved_indices_json = json.load(f)
    retrieved_indices = {}
    for row_idx, val_dict in retrieved_indices_json.items():
        retrieved_indices[int(row_idx)] = {int(col_idx): list_of_tups for col_idx, list_of_tups in val_dict.items()}
    myprint(f'<loaded from {local_path}/{filename} and integer-ified>')
    
    return retrieved_indices

def load_all_row_idxs_lookup(task, local_path, optional_suffix=''):
    with open(f"{local_path}/all_row_idxs_lookup{optional_suffix}.json", 'r') as f:
        all_row_idxs_json = json.load(f)
    all_row_idxs_lookup = {}
    for k, v in all_row_idxs_json.items():
        all_row_idxs_lookup[int(k)] = v
    return all_row_idxs_lookup

def is_png_img(item):
    return isinstance(item, PngImagePlugin.PngImageFile)

def get_all_row_idxs_for_1M_states(task):
    if task.startswith('babyai'):
        with open("dataset_jat_regent/all_row_idxs_babyai_1000000.json", "r") as f:
            all_row_idxs = json.load(f)
        return all_row_idxs[task]
    elif task.startswith('atari'): # warning only about 450k states per env
        last_row_idx = {'atari-alien': 14134, 'atari-amidar': 14319, 'atari-assault': 14427, 'atari-asterix': 14456, 'atari-asteroids': 14348, 'atari-atlantis': 14325, 'atari-bankheist': 14167, 'atari-battlezone': 13981, 'atari-beamrider': 13442, 'atari-berzerk': 13534, 'atari-bowling': 14110, 'atari-boxing': 14542, 'atari-breakout': 13474, 'atari-centipede': 14196, 'atari-choppercommand': 13397, 'atari-crazyclimber': 14026, 'atari-defender': 13504, 'atari-demonattack': 13499, 'atari-doubledunk': 14292, 'atari-enduro': 13260, 'atari-fishingderby': 14073, 'atari-freeway': 14016, 'atari-frostbite': 14075, 'atari-gopher': 13143, 'atari-gravitar': 14405, 'atari-hero': 14044, 'atari-icehockey': 14017, 'atari-jamesbond': 12678, 'atari-kangaroo': 14248, 'atari-krull': 14204, 'atari-kungfumaster': 14030, 'atari-montezumarevenge': 14219, 'atari-mspacman': 14120, 'atari-namethisgame': 13575, 'atari-phoenix': 13539, 'atari-pitfall': 14287, 'atari-pong': 14151, 'atari-privateeye': 14105, 'atari-qbert': 14026, 'atari-riverraid': 14275, 'atari-roadrunner': 14127, 'atari-robotank': 14079, 'atari-seaquest': 14097, 'atari-skiing': 14708, 'atari-solaris': 14199, 'atari-spaceinvaders': 12652, 'atari-stargunner': 13822, 'atari-surround': 13840, 'atari-tennis': 14062, 'atari-timepilot': 13896, 'atari-tutankham': 13121, 'atari-upndown': 13504, 'atari-venture': 14260, 'atari-videopinball': 14272, 'atari-wizardofwor': 13920, 'atari-yarsrevenge': 13981, 'atari-zaxxon': 13833}
        return list(range(last_row_idx[task]))
    elif task.startswith('metaworld'):
        last_row_idx = 10000
        return list(range(last_row_idx))
    else:
        last_row_idx = {'mujoco-ant': 4023, 'mujoco-doublependulum': 4002, 'mujoco-halfcheetah': 4000, 'mujoco-hopper': 4931, 'mujoco-humanoid': 4119, 'mujoco-pendulum': 4959, 'mujoco-pusher': 9000, 'mujoco-reacher': 9000, 'mujoco-standup': 4000, 'mujoco-swimmer': 4000, 'mujoco-walker': 4101}
        return list(range(last_row_idx[task]))

def get_all_row_idxs_for_100k_states(task):
    if task.startswith('babyai'):
        with open("dataset_jat_regent/all_row_idxs_babyai_100000.json", "r") as f:
            all_row_idxs = json.load(f)
        return all_row_idxs[task]
    elif task.startswith('atari'):
        last_row_idx = {'atari-alien': 3135, 'atari-amidar': 3142, 'atari-assault': 3132, 'atari-asterix': 3181, 'atari-asteroids': 3127, 'atari-atlantis': 3128, 'atari-bankheist': 3156, 'atari-battlezone': 3136, 'atari-beamrider': 3131, 'atari-berzerk': 3127, 'atari-bowling': 3148, 'atari-boxing': 3227, 'atari-breakout': 3128, 'atari-centipede': 3176, 'atari-choppercommand': 3144, 'atari-crazyclimber': 3134, 'atari-defender': 3127, 'atari-demonattack': 3127, 'atari-doubledunk': 3175, 'atari-enduro': 3126, 'atari-fishingderby': 3155, 'atari-freeway': 3131, 'atari-frostbite': 3146, 'atari-gopher': 3128, 'atari-gravitar': 3202, 'atari-hero': 3144, 'atari-icehockey': 3138, 'atari-jamesbond': 3131, 'atari-kangaroo': 3160, 'atari-krull': 3162, 'atari-kungfumaster': 3143, 'atari-montezumarevenge': 3168, 'atari-mspacman': 3143, 'atari-namethisgame': 3131, 'atari-phoenix': 3127, 'atari-pitfall': 3131, 'atari-pong': 3160, 'atari-privateeye': 3158, 'atari-qbert': 3136, 'atari-riverraid': 3157, 'atari-roadrunner': 3150, 'atari-robotank': 3133, 'atari-seaquest': 3138, 'atari-skiing': 3271, 'atari-solaris': 3129, 'atari-spaceinvaders': 3128, 'atari-stargunner': 3129, 'atari-surround': 3143, 'atari-tennis': 3129, 'atari-timepilot': 3132, 'atari-tutankham': 3127, 'atari-upndown': 3127, 'atari-venture': 3148, 'atari-videopinball': 3130, 'atari-wizardofwor': 3138, 'atari-yarsrevenge': 3129, 'atari-zaxxon': 3133}
        return list(range(last_row_idx[task]))
    elif task.startswith('metaworld'):
        last_row_idx = 1000
        return list(range(last_row_idx))
    else:
        last_row_idx = {'mujoco-ant': 401, 'mujoco-doublependulum': 401, 'mujoco-halfcheetah': 400, 'mujoco-hopper': 491, 'mujoco-humanoid': 415, 'mujoco-pendulum': 495, 'mujoco-pusher': 1000, 'mujoco-reacher': 2000, 'mujoco-standup': 400, 'mujoco-swimmer': 400, 'mujoco-walker': 407}
        return list(range(last_row_idx[task]))
    
def get_num_demos_from_num_states(task, num_states):
    local_path = f"dataset_jat_regent/{task}"
    with open(f"{local_path}/ep_lens.json", 'r') as f:
        ep_lens = json.load(f)
    ep_lens = {int(k): v for k, v in ep_lens.items()}

    # add up ep_lens of demos until num_states is reached
    num_demos = 0
    num_states_so_far = 0
    for ep, ep_len in ep_lens.items():
        num_demos += 1
        num_states_so_far += min(ep_len, 200*32) # either ep_len or 200 rows of 32 states because we limit each episode to 200 rows
        if num_states_so_far >= num_states:
            break

    myprint(f'>>We have {num_states=} ==> {num_demos=} with exact {num_states_so_far=}.<<')

    return num_demos
    
def get_all_row_idxs_for_num_demos(task, num_demos):
    local_path = f"dataset_jat_regent/{task}"
    with open(f"{local_path}/eps_2_rows_tokenized.json", 'r') as f:
        eps_2_rows_tokenized = json.load(f)
    eps_2_rows_tokenized = {int(k): v for k, v in eps_2_rows_tokenized.items()}

    if task.startswith("babyai"):
        with open(f"{local_path}/text_2_eps.json", 'r') as f:
            text_2_eps = json.load(f)
        with open(f"dataset_jat_regent/all_train_missions_babyai_100000.json", 'r') as f: # will have atleast 20 episodes
            all_train_missions = json.load(f)
        # with open(f"dataset_jat_regent/all_unseen_missions_babyai_100000.json", 'r') as f: # will have atleast 20 episodes
        #     all_unseen_missions = json.load(f)
        all_eps = {}
        all_row_idxs = {}
        num_eps_avail_in_missions = {}
        for mission in all_train_missions[task]:
            num_eps_avail = len(text_2_eps[mission])
            num_eps_avail_in_missions[mission] = num_eps_avail
            if num_eps_avail < num_demos and task != 'babyai-move-two-across-s8n9':
                continue

            all_eps[mission] = text_2_eps[mission][:num_demos]
            all_row_idxs[mission] = []
            for ep in all_eps[mission]:
                all_row_idxs[mission] += eps_2_rows_tokenized[ep]

        assert len(all_row_idxs) > 0, f'No mission has atleast {num_demos} demos. Reduce num_demos! {num_eps_avail_in_missions=}'
    else:
        all_eps = list(eps_2_rows_tokenized.keys())[:num_demos]
        all_row_idxs = []
        for ep in all_eps:
            all_row_idxs += eps_2_rows_tokenized[ep]

    if task.startswith("atari"):
        all_row_idxs = all_row_idxs[:200 * num_demos] # restricting to the first (200 * num_demos) rows in the dataset

    return all_row_idxs

def get_obs_dim(task):
    assert task.startswith("babyai") or task.startswith("metaworld") or task.startswith("mujoco")

    all_obs_dims={'mujoco-ant': 27, 'mujoco-doublependulum': 11, 'mujoco-halfcheetah': 17, 'mujoco-hopper': 11, 'mujoco-humanoid': 376, 'mujoco-pendulum': 4, 'mujoco-pusher': 23, 'mujoco-reacher': 11, 'mujoco-standup': 376, 'mujoco-swimmer': 8, 'mujoco-walker': 17}
    
    if task.startswith("babyai"):
        return (148,) # OG: 212 # removing last 64 text tokens
    elif task.startswith("metaworld"):
        return (39,)
    else:
        return (all_obs_dims[task],)

def get_act_dim(task):
    assert task.startswith("babyai") or task.startswith("metaworld") or task.startswith("mujoco")

    if task.startswith("babyai"):
        return 1
    elif task.startswith("metaworld"):
        return 4
    elif task.startswith("mujoco"):
        all_act_dims={'mujoco-ant': 8, 'mujoco-doublependulum': 1, 'mujoco-halfcheetah': 6, 'mujoco-hopper': 3, 'mujoco-humanoid': 17, 'mujoco-pendulum': 1, 'mujoco-pusher': 7, 'mujoco-reacher': 2, 'mujoco-standup': 17, 'mujoco-swimmer': 2, 'mujoco-walker': 6}
        return all_act_dims[task]
    
def get_max_abs_actions(task):

    if task.startswith("metaworld"):
        max_abs_actions = np.array([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
    elif task.startswith("mujoco"):
        max_abs_actions_dict = {'mujoco-ant': np.array([1.3034091 , 1.21738696, 1.80507922, 1.26957405, 0.96434045,
        1.04654288, 1.73047996, 1.03044176]), 'mujoco-doublependulum': np.array([0.88132131]), 'mujoco-halfcheetah': np.array([2.60063934, 2.58294296, 2.36154079, 2.18251514, 2.26374197,
        2.26263571]), 'mujoco-hopper': np.array([2.38772511, 3.72189355, 3.44262028]), 'mujoco-humanoid': np.array([2.74703884, 3.02014732, 2.86938906, 2.69342232, 2.84977007,
        2.79250097, 2.83854032, 3.37679338, 2.60311651, 3.10195351,
        2.05592704, 2.71147895, 2.86133552, 2.80322409, 3.00923967,
        2.73383904, 2.55132008]), 'mujoco-pendulum': np.array([0.33774364]), 'mujoco-pusher': np.array([1.95210719, 1.61977971, 1.21499324, 2.70512152, 1.15703058,
        1.29327738, 1.29735947]), 'mujoco-reacher': np.array([0.50809288, 0.39707077]), 'mujoco-standup': np.array([8.63981152, 9.27875233, 6.65380287, 7.04437637, 8.37550545,
        6.63224983, 7.0627203 , 7.05443382, 9.94639492, 8.7010746 ,
        8.70876789, 6.78084755, 6.35247612, 6.61219311, 6.82378101,
        8.79354095, 7.30535841]), 'mujoco-swimmer': np.array([3.64596939, 2.81188512]), 'mujoco-walker': np.array([5.13491249, 4.60425568, 5.47722244, 4.92175674, 4.7684083 ,
        4.78905487])}
        max_abs_actions = max_abs_actions_dict[task].astype(np.float32)
    else:
        max_abs_actions = None
    return max_abs_actions
    
def get_dist_stats(task, optional_suffix):
    # load distances
    folder_loc = f"{HF_DATASETS_CACHE}/regent-research/regent-subset-of-jat-dataset-tokenized-local/{task}{optional_suffix}"
    with open(f"{folder_loc}/shapes.json", 'r') as f:
        all_shapes = json.load(f)
    all_shapes = {k: tuple(v_list) for k, v_list in all_shapes.items() if k != 'max_dist'}
    distances = np.memmap(f"{folder_loc}/distances.bin", dtype=np.float32, mode='r', shape=all_shapes['distances'])

    # compute stats
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    max_dist = np.max(distances)
    p80 = np.percentile(distances, 80)
    p85 = np.percentile(distances, 85)
    p90 = np.percentile(distances, 90)
    p95 = np.percentile(distances, 95)
    p99 = np.percentile(distances, 99)
    
    return mean_dist, std_dist, max_dist, p80, p85, p90, p95, p99

def save_images_side_by_side(list_of_channel_first_RGB_arrs_normalized, save_path):
    _, H, W = list_of_channel_first_RGB_arrs_normalized[0].shape
    N = len(list_of_channel_first_RGB_arrs_normalized)

    # transpose all images to channel last
    new_list = []
    for idx, img in enumerate(list_of_channel_first_RGB_arrs_normalized):
        assert img.shape == (3, H, W)
        new_img = img.transpose(1, 2, 0)
        new_list.append(new_img)
    
    # concatenate all images side by side
    np_arrs_side_by_side = np.concatenate(new_list, axis=1)
    assert (np_arrs_side_by_side.shape == (H, N*W, 3) and 
            np.min(np_arrs_side_by_side) >= 0.0 and 
            np.max(np_arrs_side_by_side) <= 1.0)

    # de-normalize and convert to uint8
    np_arrs_side_by_side = np.clip(np_arrs_side_by_side * 255.0, 0, 255).astype(np.uint8)
    assert (np.min(np_arrs_side_by_side) >= 0 and 
            np.max(np_arrs_side_by_side) <= 255)

    # save the image
    img = Image.fromarray(np_arrs_side_by_side, 'RGB')
    img.save(save_path)
    
def get_task_info(task):
    rew_key = 'rewards'
    attn_key = 'attention_mask'
    if task.startswith("atari"):
        obs_key = 'image_observations'
        act_key = 'discrete_actions'
        B = 32 # half of 54
        obs_dim = (512,) # embedding dim
        act_dim = 1
    elif task.startswith("babyai"):
        obs_key = 'discrete_observations' # also has 'text_observations' only for raw dataset not for tokenized dataset (as it is combined into discrete_observation in tokenized dataset)
        act_key = 'discrete_actions'
        B = 256 # half of 512
        obs_dim = get_obs_dim(task)
        act_dim = get_act_dim(task)
    elif task.startswith("metaworld") or task.startswith("mujoco"):
        obs_key = 'continuous_observations'
        act_key = 'continuous_actions'
        B = 256
        obs_dim = get_obs_dim(task)
        act_dim = get_act_dim(task)

    return rew_key, attn_key, obs_key, act_key, B, obs_dim, act_dim

def L2dist(a: np.ndarray, b: np.ndarray):
    assert a.shape == b.shape, f'inputs to L2dist must have same shape. {a.shape=} {b.shape=}'
    return np.sqrt(np.sum((a - b)**2))

def cosine_similarity(a: np.ndarray, b: np.ndarray):
    assert a.shape == b.shape, f'inputs to cosine_similarity must have same shape. {a.shape=} {b.shape=}'
    if len(a.shape) == 1:
        return np.dot(a.flatten(), b.flatten()) / (np.linalg.norm(a) * np.linalg.norm(b))
    else:
        raise NotImplementedError(f'{a.shape=}')

def process_row_of_obs_atari_full_without_mask(row_of_obs: np.ndarray | List) -> np.ndarray:

    if not isinstance(row_of_obs, np.ndarray):
        row_of_obs = np.stack([np.array(img) for img in row_of_obs])  
    assert row_of_obs.shape == (len(row_of_obs), 4, 84, 84) and isinstance(row_of_obs, np.ndarray)
    row_of_obs = row_of_obs * 0.5 + 0.5 # denormalize from [-1, 1] to [0, 1]
    row_of_obs = row_of_obs.reshape(len(row_of_obs), 4*84, 84) # put side-by-side
    row_of_obs = row_of_obs[:, np.newaxis, :, :] # add channel dimension
    assert row_of_obs.shape == (len(row_of_obs), 1, 4*84, 84)
    row_of_obs = np.repeat(row_of_obs, 3, axis=1) # repeat channel 3 times
    row_of_obs = row_of_obs.astype(np.float32) # float64 --> float32
    assert row_of_obs.shape == (len(row_of_obs), 3, 4*84, 84) # sum(attn_mask) is the batch size dimension

    return row_of_obs


def process_row_of_obs_atari_full_without_mask_with_crop(task: str, row_of_obs: np.ndarray | List) -> np.ndarray:
    cc = json.load(open(f'jat_regent/crop_coordinates.json'))

    if not isinstance(row_of_obs, np.ndarray):
        row_of_obs = np.stack([np.array(img) for img in row_of_obs])  
    assert row_of_obs.shape == (len(row_of_obs), 4, 84, 84) and isinstance(row_of_obs, np.ndarray)
    # row_of_obs = row_of_obs * 0.5 + 0.5 # denormalize from [-1, 1] to [0, 1]
    # row_of_obs = row_of_obs.reshape(len(row_of_obs), 4*84, 84) # put side-by-side
    # row_of_obs = row_of_obs[:, np.newaxis, :, :] # add channel dimension
    # assert row_of_obs.shape == (len(row_of_obs), 1, 4*84, 84)
    # row_of_obs = np.repeat(row_of_obs, 3, axis=1) # repeat channel 3 times
    # row_of_obs = row_of_obs.astype(np.float32) # float64 --> float32
    # assert row_of_obs.shape == (len(row_of_obs), 3, 4*84, 84) # sum(attn_mask) is the batch size dimension

    x1, y1, x2, y2 = cc[task]["x1"], cc[task]["y1"], cc[task]["x2"], cc[task]["y2"]
    row_of_obs = row_of_obs[:, :, y1:y2+1, x1:x2+1]

    row_of_obs = row_of_obs.reshape(len(row_of_obs), -1)

    row_of_obs = np.concatenate([row_of_obs, np.zeros((len(row_of_obs), 4*84*84 - row_of_obs.shape[1]), dtype=row_of_obs.dtype)], axis=1)

    assert row_of_obs.shape == (len(row_of_obs), 4*84*84)

    return row_of_obs

def remove_repetitions(row_of_obs):
    idxs_to_keep = []
    for idx in range(len(row_of_obs)):
        if idx == 0 or not torch.equal(row_of_obs[idx], row_of_obs[idx-1]):
            idxs_to_keep.append(idx)
    myprint(f'Removed {len(row_of_obs) - len(idxs_to_keep)} repetitions.')
    row_of_obs = row_of_obs[idxs_to_keep]
    return row_of_obs

def get_emb_transform_model_dim(atari_dist_type, device, return_emb_weights=False):
    from data4robotics import load_vit, load_resnet18
    if 'vit' in atari_dist_type:
        emb_transform, emb_weights = load_vit(model_name="SOUP_1M", device=device)
        assert len(atari_dist_type.split('_')) == 2
        emb_dim = int(atari_dist_type.split('_')[1])
        emb_model = lambda batch: emb_weights(batch)[..., :emb_dim]
    elif 'resnet18' in atari_dist_type:
        emb_transform, emb_weights = load_resnet18(model_name="IN_1M_resnet18", device=device)
        assert len(atari_dist_type.split('_')) == 2
        emb_dim = int(atari_dist_type.split('_')[1])
        if emb_dim == 2048:
            emb_transform_inside = emb_transform
            emb_model = lambda batch: torch.cat( [ emb_weights(emb_transform_inside(batch[:, :, i*84:(i+1)*84, :])) for i in range(4) ] , dim=1)
            emb_transform = lambda batch: batch
        else:
            emb_model = lambda batch: emb_weights(batch)[..., :emb_dim]
    else:
        raise NotImplementedError(f'{atari_dist_type=}')
    emb_dim_tup = (emb_dim,)
    
    if return_emb_weights:
        return emb_transform, emb_model, emb_dim_tup, emb_weights
    return emb_transform, emb_model, emb_dim_tup

def collect_all_atari_data(task, dataset, all_row_idxs=None, return_atari_datarows=False, atari_dist_type='resnet18_512'):
    if all_row_idxs is None:
        all_row_idxs = list(range(len(dataset['train'])))

    # get embedding model, device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    emb_transform, emb_model, emb_dim = get_emb_transform_model_dim(atari_dist_type, device)
    
    # collect data
    all_rows_of_obs = []
    all_attn_masks = []
    if return_atari_datarows:
        all_atari_datarows = {}
    for count, row_idx in enumerate(all_row_idxs):
        datarow = dataset['train'][row_idx]
        row_of_obs = process_row_of_obs_atari_full_without_mask(datarow['image_observations'])
        row_of_obs = torch.from_numpy(row_of_obs).to(device)
        with torch.no_grad():
            row_of_obs = emb_model(emb_transform(row_of_obs)).cpu().numpy()
        attn_mask = np.array(datarow['attention_mask']).astype(bool)
        all_rows_of_obs.append(row_of_obs) # appending tensor
        all_attn_masks.append(attn_mask) # appending np array

        if count % max(len(all_row_idxs) // 10, 1) == 0 or count == len(all_row_idxs) - 1:
            myprint(f'Loaded: {count+1} / {len(all_row_idxs)} rows')

        if return_atari_datarows:
            all_atari_datarows[row_idx] = datarow

    all_rows_of_obs = np.stack(all_rows_of_obs, axis=0)
    all_attn_masks = np.stack(all_attn_masks, axis=0)
    assert (all_rows_of_obs.shape == (len(all_row_idxs), 32, *emb_dim) and
            all_attn_masks.shape == (len(all_row_idxs), 32))
    
    if return_atari_datarows:
        return all_attn_masks, all_rows_of_obs, all_atari_datarows

    return all_attn_masks, all_rows_of_obs, None

def collect_all_data(dataset, task, obs_key, num_demos, return_datarows_dict=False, atari_dist_type='resnet18_512'):
    myprint(f'<Collecting all {"Image" if task.startswith("atari") else "Vector"} observations and attention masks (all np arrays)>')

    # rows
    all_row_idxs = get_all_row_idxs_for_num_demos(task, num_demos)
    
    # collect
    if task.startswith("atari"):
        all_attn_masks_OG, all_rows_of_obs_OG, datarows_dict = collect_all_atari_data(task, dataset, all_row_idxs, return_atari_datarows=return_datarows_dict, atari_dist_type=atari_dist_type)
    elif task.startswith("babyai"):
        all_rows_of_obs_OG = {}
        all_attn_masks_OG = {}
        if return_datarows_dict:
            datarows_dict = {}
        for mission_idx, mission in enumerate(all_row_idxs.keys()):
            myprint(('*'*50) + f'{mission=} - {mission_idx+1}/{len(all_row_idxs.keys())}')
            datarows_mission = dataset['train'][all_row_idxs[mission]]
            if return_datarows_dict:
                datarows_dict[mission] = {row_idx: dataset['train'][row_idx] for row_idx in all_row_idxs[mission]}
            all_rows_of_obs_OG[mission] =  np.array(datarows_mission[obs_key]).astype(np.float32)[:, :, :148] # float64 --> float32, # removing last 64 text tokens (212 --> 148)
            all_attn_masks_OG[mission] = np.array(datarows_mission['attention_mask']).astype(bool)
    else:
        datarows = dataset['train'][all_row_idxs]
        if return_datarows_dict:
            datarows_dict = {row_idx: dataset['train'][row_idx] for row_idx in all_row_idxs}
        all_rows_of_obs_OG = np.array(datarows[obs_key]).astype(np.float32) # float64 --> float32
        all_attn_masks_OG = np.array(datarows['attention_mask']).astype(bool)
    
    # asserts
    if task.startswith("babyai"):
        for mission in all_row_idxs.keys():
            assert (isinstance(all_rows_of_obs_OG[mission], np.ndarray) and 
                    isinstance(all_attn_masks_OG[mission], np.ndarray) and 
                    isinstance(all_row_idxs[mission], list))
            myprint(f'{mission=} {all_rows_of_obs_OG[mission].shape=}, {all_attn_masks_OG[mission].shape=}.')
    else:
        assert (isinstance(all_rows_of_obs_OG, np.ndarray) and 
                isinstance(all_attn_masks_OG, np.ndarray) and 
                isinstance(all_row_idxs, list))
        myprint(f'{all_rows_of_obs_OG.shape=}, {all_attn_masks_OG.shape=}.')
    
    # return
    myprint(f'</Collecting all {"Image" if task.startswith("atari") else "Vector"} observations and attention masks (all np arrays)>')
    
    if return_datarows_dict:
        return all_rows_of_obs_OG, all_attn_masks_OG, all_row_idxs, datarows_dict
    
    return all_rows_of_obs_OG, all_attn_masks_OG, all_row_idxs, None

def process_obs_and_collect_indices(all_rows_of_obs_OG, 
                                    all_attn_masks_OG, 
                                    all_row_idxs, 
                                    kwargs
    ):
    """
    Function to collect subset of data given all_row_idxs, reshape it, create all_indices and return.
    Used in both retrieve_atari() and retrieve_vector() --> build_index_vector().
    """
    myprint(f'<Processing obs and collecting indices (all np arrays)>')
    # read kwargs
    B, task, obs_dim, device = kwargs['B'], kwargs['task'], kwargs['obs_dim'], kwargs['device']
    assert (all_rows_of_obs_OG.shape == (len(all_row_idxs), B, *obs_dim) and
            all_attn_masks_OG.shape == (len(all_row_idxs), B))

    # process: reshape and take attn mask subset
    all_attn_masks = all_attn_masks_OG.reshape(-1)
    all_processed_rows_of_obs = all_rows_of_obs_OG.reshape(-1, *obs_dim)
    all_processed_rows_of_obs = all_processed_rows_of_obs[all_attn_masks]
    assert (all_attn_masks.shape == (len(all_row_idxs) * B,) and 
            all_processed_rows_of_obs.shape == (np.sum(all_attn_masks), *obs_dim))

    # collect indices of data
    all_indices = np.array([[row_idx, i] for row_idx in all_row_idxs for i in range(B)])
    all_indices = all_indices[all_attn_masks] # this is fine because all attn masks have 0s that only come after 1s
    assert all_indices.shape == (np.sum(all_attn_masks), 2)

    myprint(f'{all_indices.shape=}, {all_processed_rows_of_obs.shape=}.')
    myprint(f'</Processing obs and collecting indices (all np arrays)>')
    return all_indices, all_processed_rows_of_obs

def retrieve_atari(row_of_obs, # query: (xdim, *obs_dim)
            all_processed_rows_of_obs,
            all_indices,
            num_to_retrieve,
            kwargs
    ):
    """
    Retrieval for Atari with images, ssim distance, and on GPU.
    """
    myprint(f'<Atari retrieval>')
    # read kwargs # Note: B = len of row
    B, device, batch_size_retrieval, obs_dim, task, verbose = kwargs['B'], kwargs['device'], kwargs['batch_size_retrieval'], kwargs['obs_dim'], kwargs['task'], kwargs['verbose']

    # make tensor on GPU
    if isinstance(row_of_obs, np.ndarray):
        row_of_obs = torch.from_numpy(row_of_obs).to(device)
    assert isinstance(row_of_obs, torch.Tensor) and row_of_obs.is_cuda
    assert isinstance(all_processed_rows_of_obs, torch.Tensor) and all_processed_rows_of_obs.is_cuda
    
    # batch size of row_of_obs which can be <= B since we process before calling this function
    xdim = row_of_obs.shape[0]

    # collect subset of data that we can retrieve from
    ydim = all_processed_rows_of_obs.shape[0]

    # iterate over data that we can retrieve from in batches
    all_scores = []
    batch_size_input = xdim
    for i in range(0, xdim, batch_size_input):
        # first argument for ssim
        xbatch_og = row_of_obs[i:i+batch_size_input]
        xbdim = xbatch_og.shape[0]
        ### t0 = time.time()
        xbatch = xbatch_og.repeat_interleave(batch_size_retrieval, dim=0)
        ### myprint(f'repeat interleave: time: {time.time() - t0}')
        assert xbatch.shape == (xbdim * batch_size_retrieval, *obs_dim)

        batch_scores = []
        for j in range(0, ydim, batch_size_retrieval):
            # second argument for ssim
            ybatch = all_processed_rows_of_obs[j:j+batch_size_retrieval]
            ybdim = ybatch.shape[0]
            repeat_pattern = (xbdim, 1, 1, 1) if task.startswith("atari") else (xbdim, 1)
            ### t0 = time.time()
            ybatch = ybatch.repeat(*repeat_pattern)
            ### myprint(f'repeat: time: {time.time() - t0}')
            assert ybatch.shape == (ybdim * xbdim, *obs_dim)

            if ybdim < batch_size_retrieval: # for last batch
                xbatch = xbatch_og.repeat_interleave(ybdim, dim=0)
            assert xbatch.shape == (xbdim * ybdim, *obs_dim)

            # compare via score_fn and update all_scores
            score_fn = lambda x, y: ssim(x, y, data_range=1.0, size_average=False) if task.startswith("atari") else F.mse_loss(x, y, reduction='none').mean(dim=1)
            ### t0 = time.time()
            score = score_fn(xbatch, ybatch)
            ### myprint(f'ssim: time: {time.time() - t0}')
            score = score.reshape(xbdim, ybdim)
            batch_scores.append(score)

        # concat
        batch_scores = torch.cat(batch_scores, dim=1)
        assert batch_scores.shape == (xbdim, ydim)
        all_scores.append(batch_scores)

    # concat
    all_scores = torch.cat(all_scores, dim=0)
    assert all_scores.shape == (xdim, ydim)

    assert all_indices.shape == (ydim, 2)

    # get top-k indices
    topk_values, topk_indices = torch.topk(all_scores, num_to_retrieve, dim=1, largest=True)
    topk_indices = topk_indices.cpu().numpy()
    assert topk_indices.shape == (xdim, num_to_retrieve)

    # convert topk indices to indices in the dataset
    retrieved_indices = all_indices[topk_indices]
    assert retrieved_indices.shape == (xdim, num_to_retrieve, 2)

    if verbose: myprint(f'{topk_indices=}, {retrieved_indices=}')

    myprint(f'</Atari retrieval>')
    return retrieved_indices

def build_index_vector(all_rows_of_obs_OG, 
                       all_attn_masks_OG, 
                       all_row_idxs, 
                       kwargs
    ):
    """
    Builds FAISS index for vector observation environments.
    """
    # read kwargs # Note: B = len of row
    nb_cores_autofaiss = kwargs['nb_cores_autofaiss']

    # take subset based on all_row_idxs, reshape, and save indices of data
    all_indices, all_processed_rows_of_obs = process_obs_and_collect_indices(all_rows_of_obs_OG=all_rows_of_obs_OG,
                                                                            all_attn_masks_OG=all_attn_masks_OG,
                                                                            all_row_idxs=all_row_idxs,
                                                                            kwargs=kwargs)

    # need to normalize for cosine
    # all_processed_rows_of_obs = all_processed_rows_of_obs / np.linalg.norm(all_processed_rows_of_obs, axis=1, keepdims=True)

    # build index
    myprint(f'\n<Building index>\n')
    knn_index, knn_index_infos = build_index(embeddings=all_processed_rows_of_obs, # Note: embeddings have to be float to avoid errors in autofaiss / embedding_reader!
                                            save_on_disk=False,
                                            min_nearest_neighbors_to_retrieve=20, # default: 20
                                            max_index_query_time_ms=10, # default: 10
                                            max_index_memory_usage="25G", # default: "16G"
                                            current_memory_available="50G", # default: "32G"
                                            metric_type='l2',
                                            nb_cores=nb_cores_autofaiss, # default: None # "The number of cores to use, by default will use all cores" as seen in https://criteo.github.io/autofaiss/getting_started/quantization.html#the-build-index-command
                                            )
    myprint(f'\n</Building index>\n')
    return all_indices, knn_index

def retrieve_vector(row_of_obs, # query: (xdim, *obs_dim)
            knn_index,
            all_indices,
            num_to_retrieve,
            kwargs
    ):
    """
    Retrieval for vector observation environments.
    """
    myprint(f'<Vector retrieval>')
    assert isinstance(row_of_obs, np.ndarray)

    # read few kwargs
    B, task = kwargs['B'], kwargs['task']

    # batch size of row_of_obs which can be <= B since we process before calling this function
    xdim = row_of_obs.shape[0]

    # # retrieve (seq)
    # topk_indices = []
    # topk_distances = []
    # for idx in range(0, xdim):
    #     tkd, tki = knn_index.search(row_of_obs[idx:idx+1], 10 * num_to_retrieve)
    #     topk_distances.append(tkd)
    #     topk_indices.append(tki)
    # topk_distances = np.concatenate(topk_distances)
    # topk_indices = np.concatenate(topk_indices)

    # retrieve (parallel)
    topk_distances, topk_indices = knn_index.search(row_of_obs, 10 * num_to_retrieve)

    topk_indices = topk_indices.astype(int)
    assert topk_indices.shape == (xdim, 10 * num_to_retrieve)
    assert topk_distances.shape == (xdim, 10 * num_to_retrieve)
    for dist_row_idx, dist_row in enumerate(topk_distances):
        assert np.all(np.diff(dist_row) >= 0), f'topk distances not in ascending order: {dist_row_idx} row'

    # remove -1s and crop to num_to_retrieve
    try:
        topk_indices = np.array([[idx for idx in indices if idx != -1][:num_to_retrieve] for indices in topk_indices])
    except:
        print(f'---------------------------------------------------Too many -1s from topk_indices ----------------------------------------------------')
        temp_topk_indices = [[idx for idx in indices if idx != -1][:num_to_retrieve] for indices in topk_indices]
        # print(f'topk_indices after removing -1s and cropping is {temp_topk_indices}')
        print(f'after -1s, min len: {min([len(indices) for indices in temp_topk_indices])}, max len {max([len(indices) for indices in temp_topk_indices])}')
        print(f'-------------------------------------------------------------------------------------------------------------------------------------------')
        print(f'Leaving some -1s in topk_indices and continuing')
        topk_indices = np.array([row+[-1 for _ in range(num_to_retrieve-len(row))] for row in temp_topk_indices])
    
    # if babyai and if not enough indices, pad with -1s
    if topk_indices.shape[1] < num_to_retrieve:
        myprint(f'******************************************Padding with -1s******************************************')
        if task.startswith("babyai"):
            myprint(f'Reason: not enough data to retrieve from. {topk_indices.shape[1]=} / {num_to_retrieve=}')
        else:
            myprint(f'Reason: not dense enough index. {topk_indices.shape[1]=} / {num_to_retrieve=}')
        topk_indices = np.concatenate([topk_indices, -1 * np.ones((xdim, num_to_retrieve - topk_indices.shape[1]), dtype=int)], axis=1)
        myprint(f'************************************************************************************')

    assert topk_indices.shape == (xdim, num_to_retrieve), f'{topk_indices.shape=}, {(xdim, num_to_retrieve)=}'

    # convert topk indices to indices in the dataset
    retrieved_indices = all_indices[topk_indices]
    assert retrieved_indices.shape == (xdim, num_to_retrieve, 2)

    myprint(f'</Vector retrieval>')
    return retrieved_indices

def convert_state_to_image(state: List[float], task: str) -> np.ndarray:
    env_kwargs = {"render_mode": "rgb_array"}
    env = make_seen_and_unseen(task, **env_kwargs)
    init_obs, _ = env.reset()
    qpos_len, qvel_len = len(env.unwrapped.init_qpos), len(env.unwrapped.init_qvel)

    # Set the state of the environment to the state
    qpos_len = len(env.unwrapped.init_qpos)
    qvel_len = len(env.unwrapped.init_qvel)
    if task in ['mujoco-halfcheetah', 'mujoco-hopper']:
        # For halfcheetah,
        #   See: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/half_cheetah_v4.py#L99
        #   What is excluded is x coord of front tip. See: https://gymnasium.farama.org/environments/mujoco/half_cheetah/
        # For hopper,
        #   See: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/hopper_v4.py#L121
        #   For excluded, see: https://gymnasium.farama.org/environments/mujoco/hopper/
        assert len(state) == qpos_len - 1 + qvel_len
        qpos, qvel = state[:qpos_len - 1], state[qpos_len - 1:]
        qpos = [env.unwrapped.init_qpos[0]] + qpos
        # print(f'prepended value: {env.unwrapped.init_qpos[0]}')
    elif task in ['mujoco-ant', 'mujoco-humanoid', 'mujoco-standup', 'mujoco-swimmer']:
        # For ant,
        #   See: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/ant_v4.py#L165
        #   For excluded, see: https://gymnasium.farama.org/environments/mujoco/ant/
        # For humanoid,
        #   See: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/humanoid_v4.py#L108
        # For standup,
        #   See: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/humanoidstandup_v4.py#L40
        # For swimmer,
        #   See: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/swimmer_v4.py#L91
        qpos, qvel = state[:qpos_len - 2], state[qpos_len - 2:qpos_len - 2 + qvel_len]
        qpos = [env.unwrapped.init_qpos[0], env.unwrapped.init_qpos[1]] + qpos
    elif task.startswith('metaworld'):
        # TODO:
        raise NotImplementedError
    else:
        raise NotImplementedError

    qpos, qvel = np.array(qpos), np.array(qvel)
    env.unwrapped.set_state(qpos, qvel)

    # render state as np.uint8 image
    img = np.array(env.render(), dtype=np.uint8)
    return img

def get_images_of_retrieved_obs(all_retrieved_obs, all_retrieved_rew, all_retrieved_act, task):
    rew_key, attn_key, obs_key, act_key, B, obs_dim, act_dim = get_task_info(task)

    if task.startswith("atari"): # if atari images retrieved
        all_retrieved_obs = np.stack(all_retrieved_obs).astype(np.float32)
        if len(all_retrieved_obs.shape) == 4:
            all_retrieved_obs = all_retrieved_obs[:, np.newaxis, :, :, :]
        num_envs, num_contexts = all_retrieved_obs.shape[0], all_retrieved_obs.shape[1] + 1
        assert all_retrieved_obs.shape == (num_envs, num_contexts - 1, 4, 84, 84)
        all_retrieved_obs_images = all_retrieved_obs * 0.5 + 0.5 # denormalize from [-1, 1] to [0, 1]
        all_retrieved_obs_images = (all_retrieved_obs_images * 255).astype(np.uint8) # denormalize from [0, 1] to [0, 255] and make into uint8
        assert all_retrieved_obs_images.shape == (num_envs, num_contexts - 1, 4, 84, 84)

        all_retrieved_act = np.stack(all_retrieved_act).astype(np.int32)
        all_retrieved_rew = np.stack(all_retrieved_rew).astype(np.float32)
        assert all_retrieved_act.shape == (num_envs, num_contexts - 1)
        assert all_retrieved_rew.shape == (num_envs, num_contexts - 1)
        
        all_retrieved_images = (all_retrieved_obs_images, all_retrieved_rew, all_retrieved_act)
    elif task.startswith("mujoco"):
        # assert len(all_retrieved_obs.shape) == 3 # (num_envs, num_retrieved, *obs_dim)
        all_retrieved_images = np.stack([np.stack([convert_state_to_image(list(state), task) for state in row_of_states]) for row_of_states in all_retrieved_obs])
    elif task.startswith("metaworld"):
        all_retrieved_obs = np.stack(all_retrieved_obs).astype(np.float32)
        num_envs, num_contexts = all_retrieved_obs.shape[0], all_retrieved_obs.shape[1] + 1
        all_retrieved_act = np.stack(all_retrieved_act).astype(np.float32)
        all_retrieved_rew = np.stack(all_retrieved_rew).astype(np.float32)
        assert all_retrieved_obs.shape == (num_envs, num_contexts - 1, *obs_dim)
        assert all_retrieved_act.shape == (num_envs, num_contexts - 1, act_dim)
        assert all_retrieved_rew.shape == (num_envs, num_contexts - 1)
        
        all_retrieved_images = (all_retrieved_obs, all_retrieved_rew, all_retrieved_act)
    else:
        raise NotImplementedError(f'{task=} not implemented yet! Only atari and mujoco and metaworld are implemented!')
    return all_retrieved_images

def atari_plotter(task, rendered_frame, all_observations, all_list_rewards, all_actions, retrieved_content, idx, step, eval_args):
    # render env
    cur_render = rendered_frame

    # separate retrieved content
    retrieved_images, retrieved_rew, retrieved_act = retrieved_content

    # get query stuff
    cur_img = all_observations[idx]["image_observation"] # (84, 84, 4)
    cur_img = cur_img.transpose(2, 0, 1)[np.newaxis, ...] # (1, 4, 84, 84)
    cur_img = cur_img.reshape(1, 1, 4*84, 84).repeat(3, axis=1)
    prev_rew = all_list_rewards[idx][-2] if len(all_list_rewards[idx]) >= 2 else 0
    cur_act = convert_local_to_global_action(all_actions[idx], task)

    # get retrieved stuff
    cur_retrieved_imgs = retrieved_images[idx] # (num_contexts - 1 , 4, 84, 84)
    cur_retrieved_imgs = cur_retrieved_imgs.reshape(len(cur_retrieved_imgs), 1, 4*84, 84).repeat(3, axis=1)
    cur_retrieved_rew = retrieved_rew[idx] # (num_contexts - 1)
    cur_retrieved_act = retrieved_act[idx] # (num_contexts - 1)
    
    # plot query and retrieved stuff
    plt.figure()
    fig, axs = plt.subplots(1, len(cur_retrieved_imgs)*3+3, figsize=((len(cur_retrieved_imgs)*3+3)*2, 4*2))
    fontsize = 22
    axs[-3].imshow(cur_img[0].transpose(1, 2, 0))
    axs[-3].axis('off')
    axs[-3].set_title(r'$s^{\text{query}}$', fontsize=fontsize+12)
    axs[-2].text(0.5, 0.5, '[' + str(prev_rew) + ']', fontsize=fontsize, ha='center', va='center')
    axs[-2].axis('off')
    axs[-2].set_title(r'$r^{\text{query}}$', fontsize=fontsize+12)
    axs[-1].text(0.5, 0.5, '[' + str(cur_act) + ']', fontsize=fontsize, ha='center', va='center')
    axs[-1].axis('off')
    axs[-1].set_title(r'$a^{\text{query}}$', fontsize=fontsize+12)
    for i in range(len(cur_retrieved_imgs)):
        axs[i*3+0].imshow(cur_retrieved_imgs[i].transpose(1, 2, 0))
        axs[i*3+0].axis('off')
        axs[i*3+0].set_title(r'$s^{\text{retr}_{'+str(i)+r'}}$', fontsize=fontsize+12)
        axs[i*3+1].text(0.5, 0.5, '[' + str(cur_retrieved_rew[i]) + ']', fontsize=fontsize, ha='center', va='center')
        axs[i*3+1].axis('off')
        axs[i*3+1].set_title(r'$r^{\text{retr}_{'+str(i)+r'}}$', fontsize=fontsize+12)
        axs[i*3+2].text(0.5, 0.5, '[' + str(cur_retrieved_act[i]) + ']', fontsize=fontsize, ha='center', va='center')
        axs[i*3+2].axis('off')
        axs[i*3+2].set_title(r'$a^{\text{retr}_{'+str(i)+r'}}$', fontsize=fontsize+12)

    # save image
    imgs_save_fol = f'{eval_args.eval_fol_name}/{task}_{eval_args.num_demos}demos_{eval_args.num_episodes}eps_{eval_args.sticky_p}sticky'
    os.makedirs(imgs_save_fol, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{imgs_save_fol}/{idx}env_{step}step.png")

    # save rendered image
    image = Image.fromarray(cur_render)
    save_loc = f"{imgs_save_fol}/{idx}env_{step}step_render.png"
    image.save(save_loc)

def metaworld_plotter(task, rendered_frame, all_observations, all_list_rewards, all_actions, retrieved_content, idx, step, eval_args):    
    # render env
    cur_render = rendered_frame

    # separate retrieved content
    retrieved_obs, retrieved_rew, retrieved_act = retrieved_content

    # get query stuff
    cur_obs = all_observations[idx]["continuous_observation"]
    prev_rew = all_list_rewards[idx][-2] if len(all_list_rewards[idx]) >= 2 else 0
    cur_act = all_actions[idx]

    # get retrieved stuff
    cur_retrieved_obs = retrieved_obs[idx]
    cur_retrieved_rew = retrieved_rew[idx]
    cur_retrieved_act = retrieved_act[idx]

    # plot query and retrieved stuff
    plt.figure()
    fig, axs = plt.subplots(1, len(cur_retrieved_obs)*3+3, figsize=((len(cur_retrieved_obs)*3+3)*2, 4*2-1.5))
    fontsize = 22
    str_cur_obs_vertical = '[' + '\n'.join([str(round(cur_obs[i],3)) for i in range(11)]) + '\n.\n.\n.\n' + str(round(cur_obs[-1],3)) + ']'
    axs[-3].text(0.5, 0.5, str_cur_obs_vertical, fontsize=fontsize, ha='center', va='center')
    axs[-3].axis('off')
    axs[-3].set_title(r'$s^{\text{query}}$', fontsize=fontsize+12)
    axs[-2].text(0.5, 0.5, '[' + str(round(prev_rew,3)) + ']', fontsize=fontsize, ha='center', va='center')
    axs[-2].axis('off')
    axs[-2].set_title(r'$r^{\text{query}}$', fontsize=fontsize+12)
    str_cur_act_vertical = '[' + '\n'.join([str(round(cur_act[i],3)) for i in range(len(cur_act))]) + ']'
    axs[-1].text(0.5, 0.5, str_cur_act_vertical, fontsize=fontsize, ha='center', va='center')
    axs[-1].axis('off')
    axs[-1].set_title(r'$a^{\text{query}}$', fontsize=fontsize+12)
    for i in range(len(cur_retrieved_obs)):
        str_cur_retrieved_obs_vertical = '[' + '\n'.join([str(round(cur_retrieved_obs[i][j],3)) for j in range(11)]) + '\n.\n.\n.\n' + str(round(cur_retrieved_obs[i][-1],3)) + ']'
        axs[i*3+0].text(0.5, 0.5, str_cur_retrieved_obs_vertical, fontsize=fontsize, ha='center', va='center')
        axs[i*3+0].axis('off')
        axs[i*3+0].set_title(r'$s^{\text{retr}_{'+str(i)+r'}}$', fontsize=fontsize+12)
        axs[i*3+1].text(0.5, 0.5, '[' + str(round(cur_retrieved_rew[i],3)) + ']', fontsize=fontsize, ha='center', va='center')
        axs[i*3+1].axis('off')
        axs[i*3+1].set_title(r'$r^{\text{retr}_{'+str(i)+r'}}$', fontsize=fontsize+12)
        str_cur_retrieved_act_vertical = '[' + '\n'.join([str(round(cur_retrieved_act[i][j],3)) for j in range(len(cur_retrieved_act[i]))]) + ']'
        axs[i*3+2].text(0.5, 0.5, str_cur_retrieved_act_vertical, fontsize=fontsize, ha='center', va='center')
        axs[i*3+2].axis('off')
        axs[i*3+2].set_title(r'$a^{\text{retr}_{'+str(i)+r'}}$', fontsize=fontsize+12)

    # save image
    imgs_save_fol = f'{eval_args.eval_fol_name}/{task}_{eval_args.num_demos}demos_{eval_args.num_episodes}eps_{eval_args.sticky_p}sticky'
    os.makedirs(imgs_save_fol, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{imgs_save_fol}/{idx}env_{step}step.png")

    # save rendered image
    image = Image.fromarray(cur_render)
    save_loc = f"{imgs_save_fol}/{idx}env_{step}step_render.png"
    image.save(save_loc)

def mujoco_plotter(task, rendered_frame, all_observations, all_list_rewards, all_actions, retrieved_content, idx, step, eval_args):
    # render env
    cur_img = rendered_frame

    # plot query and retrieved images
    retrieved_images = retrieved_content
    cur_retrieved_imgs = retrieved_images[idx]
    plt.figure()
    fig, axs = plt.subplots(1, len(cur_retrieved_imgs)+1, figsize=(4*(len(cur_retrieved_imgs)+1), 4))
    axs[0].imshow(cur_img)
    axs[0].set_title('Q')
    axs[0].axis('off')
    for i in range(len(cur_retrieved_imgs)):
        axs[i+1].imshow(cur_retrieved_imgs[i])
        axs[i+1].set_title(f'R {i + 1}')
        axs[i+1].axis('off')
    plt.tight_layout()

    # save image
    imgs_save_fol = f'{eval_args.eval_fol_name}/{task}_{eval_args.num_demos}demos_{eval_args.num_episodes}eps_{eval_args.sticky_p}sticky'
    os.makedirs(imgs_save_fol, exist_ok=True)
    plt.savefig(f"{imgs_save_fol}/{idx}env_{step}step.png")

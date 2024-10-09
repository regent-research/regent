import numpy as np
from jat.eval.rl.core import TASK_NAME_TO_ENV_ID, make_atari, make_babyai, make_metaworld, make_mujoco

UNSEEN_TASK_NAME_TO_ENV_ID = {
    "atari-alien": "ALE/Alien-v5",
    "atari-mspacman": "ALE/MsPacman-v5",
    "atari-pong": "ALE/Pong-v5",
    "atari-spaceinvaders": "ALE/SpaceInvaders-v5",
    "atari-stargunner": "ALE/StarGunner-v5",
    "metaworld-bin-picking": "bin-picking-v2",
    "metaworld-box-close": "box-close-v2",
    "metaworld-door-lock": "door-lock-v2",
    "metaworld-door-unlock": "door-unlock-v2",
    "metaworld-hand-insert": "hand-insert-v2",
    "mujoco-halfcheetah": "HalfCheetah-v4",
    "mujoco-hopper": "Hopper-v4",
}

SEEN_TASK_NAME_TO_ENV_ID = {k: v for k, v in TASK_NAME_TO_ENV_ID.items() if k not in UNSEEN_TASK_NAME_TO_ENV_ID}

def get_seen_task_names():
    return list(SEEN_TASK_NAME_TO_ENV_ID.keys())

def get_unseen_task_names():
    return list(UNSEEN_TASK_NAME_TO_ENV_ID.keys())

def make_robomimic(task_name: str, **kwargs):
    raise NotImplementedError

def make_seen_and_unseen(task_name, **kwargs):
    if task_name.startswith("atari"):
        return make_atari(task_name, **kwargs)

    elif task_name.startswith("babyai"):
        return make_babyai(task_name, **kwargs)

    elif task_name.startswith("metaworld"):
        return make_metaworld(task_name, **kwargs)

    elif task_name.startswith("mujoco"):
        return make_mujoco(task_name, **kwargs)
    
    elif task_name.startswith("robomimic"):
        return make_robomimic(task_name, **kwargs)
    
    else:
        raise ValueError(f"Unknown task name: {task_name}."
                         f"\n Available seen tasks: {get_seen_task_names()}."
                         f"\n Available unseen tasks: {get_unseen_task_names()}.")
        
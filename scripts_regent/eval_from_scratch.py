#!/usr/bin/env python3
"""Eval a from-scratch model"""
import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, HfArgumentParser, AutoConfig

from jat.utils import normalize, push_to_hub, save_video_grid
from jat.processing_jat import JatProcessor
from regent.modeling_regent import JatRegentModel
from regent.modeling_RandP import RandP
from datasets import load_from_disk
from datasets.config import HF_DATASETS_CACHE
from regent.utils import myprint, L2dist, atari_plotter, mujoco_plotter, metaworld_plotter
from regent.atari_utils import _LIMITED_ACTION_SET
from copy import deepcopy
from datetime import datetime
from regent.eval.rl import make_seen_and_unseen, UNSEEN_TASK_NAME_TO_ENV_ID, SEEN_TASK_NAME_TO_ENV_ID
import matplotlib.pyplot as plt
from regent.modeling_from_scratch import ImpalaCNN, BC_MLP, process


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config we are going to train from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it "
                "will execute code present on the Hub on your local machine."
            )
        },
    )
    eval_fol_name: str = field(
        default=None,
        metadata={"help": "Path to eval folder"}
    )


@dataclass
class EvaluationArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    tasks: List[str] = field(default_factory=list, metadata={"help": "Tasks to train on."})
    use_cpu: bool = field(default=False, metadata={"help": "Use CPU instead of GPU."})
    save_video: bool = field(default=False, metadata={"help": "Save video of the evaluation."})
    save_intermediate_snapshots: bool = field(default=False, metadata={"help": "Save retrieved and query content as images in every step."})
    num_episodes: int = field(default=100, metadata={"help": "Number of episodes to evaluate on."})
    push_to_hub: bool = field(default=False, metadata={"help": "Push the model to the hub."})
    repo_id: Optional[str] = field(default=None, metadata={"help": "Repository ID to push to."})
    num_demos: int = field(default=100, metadata={"help": "Number of episodes (aka demos) to retrieve from."})
    sticky_p: float = field(default=0.0, metadata={"help": "Sticky probability (currently for atari envs only)."})
    deterministic: bool = field(default=True, metadata={"help": "Deterministic argmax of logits or multinomial sampling of logits."})


def get_default_device() -> torch.device:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def eval_rl(model: JatRegentModel, processor: JatProcessor, task, eval_args):
    # Create the environment
    env_kwargs = {}
    if task.startswith("atari"):
        env_kwargs["clip_reward"] = False
        env_kwargs["repeat_action_probability"] = eval_args.sticky_p
    if eval_args.save_video:
        env_kwargs["render_mode"] = "rgb_array"

    context_window = 32 if task.startswith("atari") else 256

    myprint(('-'*100) + f'{eval_args.num_episodes=}')
    scores = []
    num_envs = eval_args.num_episodes
    all_envs = [make_seen_and_unseen(task, **env_kwargs) for idx in range(num_envs)]
    all_seeds = [int(1e5+i) for i in range(1000)][:num_envs]
    all_observations = [all_envs[idx].reset(seed = all_seeds[idx])[0] for idx in range(num_envs)]
    
    # reset those nvs whose missions are not in the training set (and hence don't have an associated knn_index) of the babyai task
    if task.startswith("babyai"):
        for idx in range(num_envs):
            while all_observations[idx]["text_observation"] not in model.knn_index:
                myprint(f"Resetting babyai env {idx} (seed {all_seeds[idx]}) because the mission is not in the training set")
                all_observations[idx] = all_envs[idx].reset()[0]

    
    all_rewards = [None for idx in range(num_envs)]
    all_list_rewards = [[] for idx in range(num_envs)]
    all_dones = [False for idx in range(num_envs)]
    all_frames = [[] for idx in range(num_envs)]
    all_return_breakdown = [{} for idx in range(num_envs)]
    step = 0
    
    while not all(all_dones):
        all_processed = [
            process(task, processor, **all_observations[idx], reward=all_rewards[idx], action_space=all_envs[idx].action_space, context_window=context_window, deterministic=eval_args.deterministic) for idx in range(num_envs)
        ]
        if eval_args.save_intermediate_snapshots:
            all_actions, retrieved_images = model.get_next_action(all_processed, return_retrieved_obs=True)
        else:
            all_actions = model.get_next_action(all_processed)
        for idx in range(num_envs):
            if not all_dones[idx]:
                all_observations[idx], all_rewards[idx], termined, truncated, info = all_envs[idx].step(all_actions[idx])
                all_dones[idx] = termined or truncated
                
                # Handle "fake done" for atari
                if all_dones[idx] and task.startswith("atari"):
                    if "episode" not in info:
                        all_observations[idx], info = all_envs[idx].reset()
                        all_dones[idx] = False

                # Update the return
                all_list_rewards[idx].append(all_rewards[idx])

                # Render the environment and save frames
                if eval_args.save_video:
                    rendered_frame = np.array(all_envs[idx].render(), dtype=np.uint8)
                    all_frames[idx].append(rendered_frame)
                # Save query and retrieved images
                if eval_args.save_intermediate_snapshots:
                    plotter_func = atari_plotter if task.startswith("atari") else mujoco_plotter if task.startswith("mujoco") else metaworld_plotter if task.startswith("metaworld") else None
                    plotter_func(task=task, rendered_frame=rendered_frame, all_observations=all_observations, all_list_rewards=all_list_rewards, all_actions=all_actions, retrieved_content=retrieved_images, idx=idx, step=step, eval_args=eval_args)

                # Update return breakdown for unseen mujoco tasks
                if task.startswith("mujoco") and task in UNSEEN_TASK_NAME_TO_ENV_ID:
                    for k, v in info.items():
                        all_return_breakdown[idx][k] = all_return_breakdown[idx].get(k, 0) + v
        # print return stats so far
        sum_rewards_so_far = [sum(all_list_rewards[idx]) for idx in range(num_envs)]
        myprint(f"return: min {min(sum_rewards_so_far)}, max {max(sum_rewards_so_far)}, mean {np.mean(sum_rewards_so_far)}")
        
        # print return breakdown for unseen mujoco tasks
        if task.startswith("mujoco") and task in UNSEEN_TASK_NAME_TO_ENV_ID:
            fn_applied_to_rb_vals = lambda fn: {key: fn([all_return_breakdown[idx][key] for idx in range(num_envs)]) for key in all_return_breakdown[0].keys() if "reward" in key}
            myprint(f"return breakdown: min {fn_applied_to_rb_vals(min)}, max {fn_applied_to_rb_vals(max)}, mean {fn_applied_to_rb_vals(np.mean)}")

        step += 1

    for idx in range(num_envs):
        scores.append(sum(all_list_rewards[idx]))
        all_envs[idx].close()
    myprint(('-'*100) + '\n\n\n')



    raw_mean, raw_std = np.mean(scores), np.std(scores)

    # Normalize the scores
    norm_scores = normalize(scores, task, "expert")
    if norm_scores is not None:  # Can be None if random is better than expert
        norm_mean, norm_std = np.mean(norm_scores), np.std(norm_scores)
        myprint(
            f"Task {task}\t"
            f"Raw score: {raw_mean:.2f} ± {raw_std:.2f}\t"
            f"Normalized score: {norm_mean:.2f} ± {norm_std:.2f}\n"
            f"All seeds: {all_seeds}\n"
            f"All scores: {[round(s, 2) for s in scores]}\n"
            f"All normalized scores: {[round(s, 2) for s in norm_scores]}"
        )
    else:
        myprint(f"Task {task} Raw score: {raw_mean:.2f} ± {raw_std:.2f}")

    # Resize images by 1/3 to limit memory usage (the video is reduced anyway when aggregated with the others)
    if eval_args.save_video:
        import cv2

        all_frames = [[cv2.resize(frame, (0, 0), fx=1 / 3, fy=1 / 3) for frame in frames] for frames in all_frames]
        all_frames = sum(all_frames, []) # Flatten the list of lists


    return norm_scores, all_frames, all_envs[0].metadata["render_fps"]


def main():
    parser = HfArgumentParser((ModelArguments, EvaluationArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, eval_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, eval_args = parser.parse_args_into_dataclasses()
    eval_args.eval_fol_name = model_args.eval_fol_name
    myprint(f'{model_args=}')
    myprint(f'{eval_args=}')

    # if planning to save video, need to specify eval folder where it will be saved
    if eval_args.save_video and model_args.eval_fol_name is None:
        raise ValueError("Need to specify eval folder where video will be saved")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Set the tasks
    OG_tasks = deepcopy(eval_args.tasks)
    tasks = eval_args.tasks
    for domain in ["atari-seen", "babyai-seen", "metaworld-seen", "mujoco-seen"]:
        if domain in tasks:
            tasks.remove(domain)
            tasks.extend([env_id for env_id in SEEN_TASK_NAME_TO_ENV_ID.keys() if env_id.startswith(domain.split("-")[0])])

    for domain in ["atari-unseen", "metaworld-unseen", "mujoco-unseen", "robomimic-unseen"]:
        if domain in tasks:
            tasks.remove(domain)
            tasks.extend([env_id for env_id in UNSEEN_TASK_NAME_TO_ENV_ID.keys() if env_id.startswith(domain.split("-")[0])])
    
    assert len(tasks) == 1, f'Expected 1 task, got {len(tasks)} tasks: {tasks}'
    device = torch.device("cpu") if eval_args.use_cpu else get_default_device()
    
    # model
    if tasks[0].startswith("atari"):
        model = ImpalaCNN(task=tasks[0], shape=(4, 84, 84), num_actions=len(_LIMITED_ACTION_SET))
    else:
        model = BC_MLP(task=tasks[0], hidden_dims=[256, 256])
    
    model.load_state_dict(torch.load(f"{model_args.model_name_or_path}/pytorch_model.bin"))
    model = model.to(device)
    model.device = device

    processor = AutoProcessor.from_pretrained(
        'regent-research/regent-medium-embeddings', cache_dir=model_args.cache_dir, trust_remote_code=model_args.trust_remote_code
    )

    # loop over tasks
    evaluations = {}
    video_list = []
    input_fps = []

    for task in tasks:
        if task in list(SEEN_TASK_NAME_TO_ENV_ID.keys())+list(UNSEEN_TASK_NAME_TO_ENV_ID.keys()):
            myprint(('-'*100) + f'{task=}')
            dataset = load_from_disk(f'{HF_DATASETS_CACHE}/jat-project/jat-dataset-tokenized/{task}')
            scores, frames, fps = eval_rl(model, processor, task, eval_args)
            evaluations[task] = scores
            # Save the video
            if eval_args.save_video:
                video_list.append(frames)
                input_fps.append(fps)
            myprint(('-'*100) + '\n\n\n')
        else:
            warnings.warn(f"Task {task} is not supported.")

    # save scores dict
    if model_args.eval_fol_name is not None:
        os.makedirs(f"{model_args.eval_fol_name}", exist_ok=True)
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")        
        eval_path = f"{model_args.eval_fol_name}/evaluations_{current_datetime}.json"
        with open(eval_path, "a") as file:
            file.write("\n") # Add a new line to file
            json.dump(evaluations, file)
        myprint(f"Saved to {eval_path}")

    # Save the video
    if eval_args.save_video and model_args.eval_fol_name is not None:
        replay_path = f"{model_args.eval_fol_name}/replay_{current_datetime}.mp4"
        save_video_grid(video_list, input_fps, replay_path, output_fps=30, max_length_seconds=180)
        myprint(f"Saved to {replay_path}")
    else:
        replay_path = None


if __name__ == "__main__":
    main()

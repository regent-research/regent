#!/usr/bin/env python3
"""Train a JAT model"""


import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import datasets.config
from datasets import load_dataset, load_from_disk
from datasets.config import HF_DATASETS_CACHE, HF_DATASETS_OFFLINE
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, HfArgumentParser, Trainer, TrainingArguments

from regent.eval.rl.core import SEEN_TASK_NAME_TO_ENV_ID, UNSEEN_TASK_NAME_TO_ENV_ID
from jat.modeling_jat import JatModel
from regent.utils_interleave_datasets import interleave_datasets
from regent.utils import get_all_row_idxs_for_num_demos, get_all_row_idxs_for_100k_states
import torch
from peft import IA3Config, get_peft_model


# Sometimes, the server is down; increasing the number of
# retries allows to wait more instead of making the training crash
datasets.config.STREAMING_READ_MAX_RETRIES = 10000


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config we are going to train from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
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


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    tasks: List[str] = field(default_factory=list, metadata={"help": "Tasks to train on."})
    use_same_data_as_REGENT: bool = field(default=False, metadata={"help": "Use the same (much smaller) dataset as REGENT for an apples-to-apples comparison with REGENT?"})
    finetune_num_demos: int = field(default=None, metadata={"help": "Number of episodes (aka demos) to retrieve from and finetune on."})


SAMPLE_WEIGHTS = {
    "conceptual-captions": 10.0,
    "oscar": 10.0,
    "wikipedia": 10.0,
}

os.environ["WANDB_ENTITY"] = "regent-creators"
os.environ["WANDB_PROJECT"] = "jat"


class MyTrainer(Trainer):
    def _get_train_sampler(self) -> None:
        return None


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    config.use_same_data_as_REGENT = data_args.use_same_data_as_REGENT
    config.finetune_num_demos = data_args.finetune_num_demos
    model = JatModel(config)
    print(f'model = {model}')
    processor = AutoProcessor.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )

    # print total number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params/ 1e6:.4f}M")

    # load state_dict if finetuning; also change wandb project
    if config.finetune_num_demos is not None:
        state_dict_loc = f"{model_args.model_name_or_path}/pytorch_model.bin"
        model.load_state_dict(torch.load(state_dict_loc))
        print(f'loaded state_dict from {state_dict_loc}')
        os.environ["WANDB_PROJECT"] = "jat_finetune"

    # peft and print peft number of parameters
    peft_config = IA3Config(task_type="CAUSAL_LM", target_modules=["k_proj", "v_proj", "out_proj"], feedforward_modules=["out_proj"])
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Set the tasks
    tasks = data_args.tasks
    if tasks == ["all"]:
        tasks = ["atari", "babyai", "metaworld", "mujoco", "oscar", "wikipedia", "conceptual-captions", "ok-vqa"]

    for domain in ["atari", "babyai", "metaworld", "mujoco"]:
        if domain in tasks:
            tasks.remove(domain)
            tasks.extend([env_id for env_id in SEEN_TASK_NAME_TO_ENV_ID.keys() if env_id.startswith(domain)])

    # Load the datasets
    if HF_DATASETS_OFFLINE:
        for task in tasks:
            if not os.path.exists(f"{HF_DATASETS_CACHE}/jat-project/jat-dataset-tokenized/{task}"):
                raise ValueError(
                    f"""Dataset {task} not found in {HF_DATASETS_CACHE}/jat-project/jat-dataset-tokenized/
Make sure to download and save it first with
```
from datasets import load_dataset
dataset = load_dataset('jat-project/jat-dataset-tokenized', '{task}')
dataset.save_to_disk('{HF_DATASETS_CACHE}/jat-project/jat-dataset-tokenized/{task}')
```"""
                )
        
        train_dataset = {}
        for task in tqdm(tasks, desc="Loading datasets"):
            print(f'Loading {task} dataset')
            d = load_from_disk(f"{HF_DATASETS_CACHE}/jat-project/jat-dataset-tokenized/{task}")
            train_dataset[task] = d["train"]

            if config.use_same_data_as_REGENT:
                if task.startswith("babyai"):
                    all_row_idxs_query_dict = get_all_row_idxs_for_100k_states(task)
                    all_row_idxs_query = sorted(sum(list(all_row_idxs_query_dict.values()), []))
                else:
                    all_row_idxs_query = get_all_row_idxs_for_100k_states(task)
                
                train_dataset[task] = train_dataset[task].select(all_row_idxs_query)
            
            if (config.finetune_num_demos is not None and task in UNSEEN_TASK_NAME_TO_ENV_ID):
                train_dataset[task] = train_dataset[task].select(get_all_row_idxs_for_num_demos(task, config.finetune_num_demos))
    else:
        raise NotImplementedError("Online datasets are not supported yet.")

    weights = [SAMPLE_WEIGHTS.get(t, 1.0) for t in train_dataset.keys()]

    train_dataset = interleave_datasets(
        list(train_dataset.values()),
        probabilities=[w / sum(weights) for w in weights],
        seed=training_args.seed,
        stopping_strategy="all_exhausted",
        n_contiguous=training_args.per_device_train_batch_size,
    )
    print(f'------------------- len(train_dataset) = {len(train_dataset)} -------------------')

    # Due to the train dataset's structure, where every 'n' consecutive samples share the same modalities, we can't
    # load all samples at once. Different sets of 'n' samples have different modalities. Therefore, we must load and
    # process each set of 'n' samples separately.
    if training_args.dispatch_batches is not False:
        raise ValueError("Make sure to pass `--dispatch_batches False`.")

    # Why the training continue after exauhsting the dataset? https://github.com/huggingface/transformers/issues/26635
    trainer = MyTrainer(model=model, args=training_args, train_dataset=train_dataset, tokenizer=processor)
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    if config.finetune_num_demos is not None:
        # save final checkpoint when finetuning
        os.makedirs(f"{training_args.output_dir}/checkpoint-final/", exist_ok=True)
        trainer.save_model(f"{training_args.output_dir}/checkpoint-final/")


if __name__ == "__main__":
    main()

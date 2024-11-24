# REGENT: A Retrieval-Augmented Generalist Agent That Can Act in-Context In New Environments

* [REGENT on Huggingface](https://huggingface.co/regent-research)
* [REGENT checkpoint on HuggingFace](https://huggingface.co/regent-research/regent-medium-embeddings)
* [REGENT dataset on HuggingFace](https://huggingface.co/datasets/regent-research/regent-subset-of-jat-dataset-tokenized)

## Installation

```shell
conda create -n regent python=3.10
conda activate regent
pip install -e .[dev]
pip install mujoco==2.3.7 # for swimmer-v4 error
pip install pytorch_msssim autofaiss
pip install timm==0.6.11 # for vit_base embedding models
pip install git+https://github.com/facebookresearch/r3m.git # for resnet18 embedding models
```

Follow the instructions at [visual_features/resnet18/README.md](visual_features/resnet18/README.md) to download the model for embedding atari images.

## TASK: what is it?
The list of unseen tasks are the keys of the dict here: [regent/eval/rl/core.py](regent/eval/rl/core.py).
The list of all tasks (training and unseen) are the keys of the dict here: [jat/eval/rl/core.py](jat/eval/rl/core.py).

## JAT Dataset
Download the original JAT (tokenized) dataset as follows. This dataset is used for retrieval when evaluating our Retrieve and Play (R&P) and REGENT agents.
```
python scripts_regent/download_all_jat_datasets.py
```

## REGENT Dataset
Download the REGENT dataset (in parquet files) from huggingface and save as bin files locally with one command as follows. This dataset is used for computing the distance normalization value when evaluating agents. It also contains the subset of the JAT dataset used for pre-training REGENT.
```
python scripts_regent/download_all_regent_datasets_and_convert_parquet_files_to_bin_files.py
```
We provide detailed information on the creation and preprocessing of this dataset in [scripts_preprocessing/README.md](scripts_preprocessing/README.md).


## Evaluating Retrieve and Play (R&P)
Evaluate our simple 1 nearest neighbor baseline for a TASK (e.g. metaworld-door-lock), with NUM_DEMOS (e.g. 100) to retrieve from, for NUM_EPS (e.g. 100) rollouts.
```
python -u scripts_regent/eval_RandP.py \
    --tasks ${TASK} --num_demos ${NUM_DEMOS} --num_episodes ${NUM_EPS}
```
If you choose an atari-* task, please add `--sticky_p 0.05` if you'd like sticky probability in the environment.


## Evaluating REGENT 
Evaluate our REGENT checkpoint for a TASK, with NUM_DEMOS to retrieve from, for NUM_EPS rollouts.
```
python -u scripts_regent/eval_jat_regent.py \
    --model_name_or_path regent-research/regent-medium-embeddings --trust_remote_code \
    --tasks ${TASK} --num_demos ${NUM_DEMOS} --num_episodes ${NUM_EPS}
```
If you choose an atari-* task, please add `--sticky_p 0.05` if you'd like sticky probability in the environment.


## Pre-training REGENT
Pre-train REGENT on the REGENT dataset with data from all four environment suites. We use two GPUs.
```
HF_DATASETS_OFFLINE=True \
accelerate launch --main_process_port=29501 scripts_regent/train_jat_regent_tokenized.py \
--output_dir checkpoints/regent-medium-embeddings \
--model_name_or_path regent-research/regent-medium-embeddings \
--tasks atari babyai metaworld mujoco \
--run_name regent-medium-embeddings \
--save_safetensors false \
--eval_strategy epoch \
--trust_remote_code \
--per_device_train_batch_size 256 \
--gradient_accumulation_steps 1 \
--save_strategy steps \
--save_steps 1000 \
--logging_steps 100 \
--logging_first_step \
--dispatch_batches False \
--dataloader_num_workers 32 \
--num_train_epochs 3
```
We only use the checkpoint after the first epoch (usually at step 27726) for evaluation. So feel free to early stop after an epoch.


## Finetuning REGENT
Finetune REGENT on FINETUNE_NUM_DEMOS from an unseen TASK.
```
HF_DATASETS_OFFLINE=True \
accelerate launch --main_process_port=${PORT} scripts_regent/train_jat_regent_tokenized.py \
--output_dir checkpoints/finetune_${TASK}_${FINETUNE_NUM_DEMOS} \
--model_name_or_path regent-research/regent-medium-embeddings \
--tasks ${TASK} \
--finetune_num_demos ${FINETUNE_NUM_DEMOS} \
--run_name finetune_${TASK}_${FINETUNE_NUM_DEMOS} \
--save_safetensors false \
--trust_remote_code \
--per_device_train_batch_size 256 \
--gradient_accumulation_steps 1 \
--save_strategy no \
--logging_steps 10 \
--logging_first_step \
--dispatch_batches False \
--dataloader_num_workers 4 \
--num_train_epochs 3 \
--learning_rate 5e-6
```


## Pre-training JAT/Gato
Pre-train JAT/Gato on the REGENT dataset.
```
HF_DATASETS_OFFLINE=True \
accelerate launch scripts_jat_regent/train_jat_tokenized.py \
--output_dir checkpoints/jat \
--model_name_or_path jat-project/jat \
--tasks atari babyai metaworld mujoco \
--run_name train_jat \
--save_safetensors false \
--trust_remote_code \
--per_device_train_batch_size 20 \
--gradient_accumulation_steps 2 \
--save_steps 10000 \
--logging_steps 100 \
--logging_first_step \
--dispatch_batches False \
--dataloader_num_workers 16 \
--max_steps 25000 \
--use_same_data_as_REGENT true
```

Evaluate on TASK for for NUM_EPS rollouts.
```
python -u scripts_jat_regent/eval_jat.py \
--model_name_or_path checkpoints/jat/checkpoint-25000 \
--tasks ${TASK} --num_episodes ${NUM_EPS} --trust_remote_code
```
If you choose an atari-* task, please add `--sticky_p 0.05` if you'd like sticky probability in the environment.


## Parameter-Efficient Finetune (PEFT) with IA3 of JAT/Gato
PEFT with IA3 of JAT/Gato on FINETUNE_NUM_DEMOS from an unseen TASK.
```
HF_DATASETS_OFFLINE=True \
accelerate launch --main_process_port=${PORT} scripts_jat_regent/peft_jat_tokenized.py \
--output_dir checkpoints/peft_jat_${TASK}_${FINETUNE_NUM_DEMOS} \
--model_name_or_path checkpoints/jat \
--tasks ${TASK} \
--finetune_num_demos ${FINETUNE_NUM_DEMOS} \
--run_name peft_jat_${TASK}_${FINETUNE_NUM_DEMOS} \
--save_safetensors false \
--trust_remote_code \
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 1 \
--save_strategy no \
--logging_steps 10 \
--logging_first_step \
--dispatch_batches False \
--dataloader_num_workers 16 \
--max_steps 1000 \
--warmup_steps 60 \
--learning_rate 3e-3 \
--optim adafactor
```

Evaluate on TASK for for NUM_EPS rollouts.
```
python -u scripts_jat_regent/eval_jat_peft.py \
--model_name_or_path checkpoints/peft_jat_${TASK}_${FINETUNE_NUM_DEMOS} \
--tasks ${TASK} --num_episodes ${NUM_EPS} --trust_remote_code
```
If you choose an atari-* task, please add `--sticky_p 0.05` if you'd like sticky probability in the environment.


## DRIL
Please see [dril/README.md](dril/README.md).


## Citation
If you'd like to cite our work, please use:

```
@inproceedings{
anonymous2024regent,
title={{REGENT}: A Retrieval-Augmented Generalist Agent That Can Act In-Context in New Environments},
author={Anonymous},
booktitle={Submitted to The Thirteenth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=NxyfSW6mLK},
note={under review}
}
```


## Acknowledgements
This repository builds on top of the [JAT](https://huggingface.co/jat-project) project. The REGENT dataset is a subset of the JAT dataset. For both these reasons, we sincerely thank the authors of the JAT project. We also wish to thank the authors of the [data4robotics](https://github.com/SudeepDasari/data4robotics) repository.

import json
from typing import Dict, List, Optional

from huggingface_hub import HfApi, ModelCard, ModelCardData
from transformers import PreTrainedModel, ProcessorMixin
from jat.utils import generate_rl_eval_results, PRETTY_TASK_NAMES

def generate_model_card(model_name: str, evaluations: Optional[Dict[str, List[float]]] = None) -> ModelCard:
    """
    Generate a ModelCard from a template.

    Args:
        model_name (`str`):
            Model name.
        evaluations (`Dict[str, List[float]]`):
            Dictionary containing the evaluation results for each task.

    Returns:
        `ModelCard`:
            A ModelCard object.
    """
    tags = ["reinforcement-learning"]
    if evaluations is not None:
        tags.extend(evaluations.keys())
    card_data = ModelCardData(
        tags=tags,
        eval_results=generate_rl_eval_results(evaluations) if evaluations is not None else None,
        model_name=model_name,
        datasets="regent-research/regent-subset-of-jat-dataset-tokenized", # NEW:
        pipeline_tag="reinforcement-learning",
    )
    card = ModelCard.from_template(
        card_data,
        template_path="templates/model_card.md",
        model_name=model_name,
        model_id="Regent", # NEW:
        tasks=[PRETTY_TASK_NAMES[task_name] for task_name in evaluations.keys()] if evaluations is not None else [],
    )
    return card


def push_to_hub(
    model: PreTrainedModel,
    processor: ProcessorMixin,
    repo_id: str,
    replay_path: Optional[str] = None,
    eval_path: Optional[str] = None,
) -> None:
    """
    Push a model to the Hugging Face Hub.

    Args:
        model (`PreTrainedModel`):
            Model to push.
        processor (`ProcessorMixin`):
            Processor to push.
        repo_id (`str`):
            Repository ID to push to.
        replay_path (`str` or `None`, **optional**):
            Path to the replay video.
        eval_path (`str` or `None`, **optional**):
            Path to the evaluation scores.
    """
    api = HfApi()

    # Create the repo
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    # Get the evaluation scores to compute the mean and std
    if eval_path is not None:
        with open(eval_path, "r") as file:
            evaluations = json.load(file)
    else:
        evaluations = None

    # Create a README.md using a template
    model_card = generate_model_card(repo_id, evaluations)
    model_card.push_to_hub(repo_id, commit_message="Upload model card")

    # Push the model
    model.push_to_hub(repo_id, commit_message="Upload model", safe_serialization=False) # NEW: safe_serialization=False

    # Push the processor
    processor.push_to_hub(repo_id, commit_message="Upload processor")

    # Push the replay
    if replay_path is not None:
        api.upload_file(
            path_or_fileobj=replay_path,
            path_in_repo="replay.mp4",
            repo_id=repo_id,
            commit_message="Upload replay",
            repo_type="model",
        )

    # Push the evaluation scores
    if eval_path is not None:
        api.upload_file(
            path_or_fileobj=eval_path,
            path_in_repo="evaluations.json",
            repo_id=repo_id,
            commit_message="Upload evaluations",
            repo_type="model",
        )

    print(f"Pushed model to  \033[34mhttps://huggingface.co/{repo_id}\033[0m")
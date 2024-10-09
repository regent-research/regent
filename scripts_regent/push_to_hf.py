from transformers import AutoModelForCausalLM, AutoProcessor
from regent.utils_hf import push_to_hub

path = f'checkpoints/jat-regent-medium-10.0lamda-1.0MDM-1.0ADM-p95DN-resnet18_512ADT_embeddings/checkpoint-27726'
repo_id = f'regent-research/regent-medium-embeddings'

# load from path
model = AutoModelForCausalLM.from_pretrained(
    path, cache_dir=None, trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(
    path, cache_dir=None, trust_remote_code=True
)

# Push to hub
push_to_hub(model, processor, repo_id, replay_path=None, eval_path=None)

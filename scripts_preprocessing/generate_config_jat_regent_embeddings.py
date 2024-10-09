from transformers import AutoTokenizer, CLIPImageProcessor

from jat.configuration_jat import JatConfig
from jat.processing_jat import JatProcessor

def push_config_to_hf(choice):
    if choice == 'small':
        suffix = '-small-embeddings'
        hidden_size = 576
        num_layers = 12
        num_heads = 12
    elif choice == 'medium':
        suffix = '-medium-embeddings'
        hidden_size = 768
        num_layers = 12
        num_heads = 12
    elif choice == 'large':
        suffix = '-large-embeddings'
        hidden_size = 1024
        num_layers = 12
        num_heads = 16
    elif choice == 'medium512':
        suffix = '-medium512-embeddings'
        hidden_size = 512
        num_layers = 12
        num_heads = 16

    # Small model
    tokenizer = AutoTokenizer.from_pretrained("gpt2", model_input_names=["input_ids", "attention_mask"])
    config = JatConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=40, # num_contexts (see below) 20 * 2 (for state-action pairs) # OG: 512
        hidden_size=hidden_size, # OG: 768 # this needs to be divisible by num_heads (e.g. 12)
        num_layers=num_layers,
        attention_types=[[["global", "local"], 6]],
        num_heads=num_heads,
        max_discrete_value=148 + 64,  # 148 (discrete obs from BabyAI) + 64 (max size of BabyAI's text observation)
        tokenizer_class=tokenizer.__class__.__name__,
        observation_loss_coef=0.00, # No observation prediction or loss # OG: default value
        action_loss_coef=1.0,  # Only action prediction # OG: default value
        max_continuous_size=513, # OG: 377 # 512 atari emb_dim and 1 for reward
    )
    # the following originally didn't exist:
    config.action_vocab_size = 18 # actions from 0-17 in atari and 0-6 babyai 
    config.ONLY_RL_TASKS = True # only training on RL tasks data
    config.num_contexts = 20 # if you change this, you need to change the max_position_embeddings
    image_processor = CLIPImageProcessor(
        size={"shortest_edge": config.image_size}, crop_size={"height": config.image_size, "width": config.image_size}
    )
    tokenizer.model_max_length = config.max_position_embeddings
    tokenizer.pad_token = tokenizer.eos_token
    processor = JatProcessor(tokenizer=tokenizer, image_processor=image_processor)
    config.push_to_hub(f"regent-research/regent{suffix}")
    processor.push_to_hub(f"regent-research/regent{suffix}")

if __name__ == "__main__":
    for choice in ['small', 'medium', 'large', 'medium512']:
        push_config_to_hf(choice)
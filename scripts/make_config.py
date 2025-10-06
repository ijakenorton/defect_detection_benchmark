import torch
import json

# Load the checkpoint
checkpoint = torch.load('/projects/sciences/computing/norja159/research/models/pretrained/linevul/pytorch_model.bin', map_location='cpu')
state_dict = checkpoint

# Extract configuration from shapes
vocab_size = state_dict['roberta.embeddings.word_embeddings.weight'].shape[0]  # 50265
hidden_size = state_dict['roberta.embeddings.word_embeddings.weight'].shape[1]  # 768
max_position_embeddings = state_dict['roberta.embeddings.position_embeddings.weight'].shape[0]  # 514

# Count the number of layers - fixed indexing
layer_keys = [k for k in state_dict.keys() if 'roberta.encoder.layer.' in k]
if layer_keys:
    # Extract layer numbers from keys like 'roberta.encoder.layer.0.attention...'
    layer_numbers = [int(k.split('roberta.encoder.layer.')[1].split('.')[0]) for k in layer_keys]
    num_hidden_layers = max(layer_numbers) + 1
else:
    num_hidden_layers = 12  # default

# Calculate num_attention_heads
num_attention_heads = 12  # Standard for base models

# Check if there's a classification head
has_classifier = any('classifier' in k for k in state_dict.keys())

config = {
    "_name_or_path": "microsoft/codebert-base",
    "architectures": ["RobertaForSequenceClassification"] if has_classifier else ["RobertaModel"],
    "attention_probs_dropout_prob": 0.1,
    "bos_token_id": 0,
    "eos_token_id": 2,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": hidden_size,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "layer_norm_eps": 1e-05,
    "max_position_embeddings": max_position_embeddings,
    "model_type": "roberta",
    "num_attention_heads": num_attention_heads,
    "num_hidden_layers": num_hidden_layers,
    "pad_token_id": 1,
    "position_embedding_type": "absolute",
    "type_vocab_size": 1,
    "vocab_size": vocab_size
}

with open('config.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"Config created with {num_hidden_layers} layers")
print(f"Has classifier: {has_classifier}")

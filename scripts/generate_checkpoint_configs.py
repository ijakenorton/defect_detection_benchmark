#!/usr/bin/env python3
"""
Generate config.json files for existing checkpoints that don't have them.

This script:
1. Scans checkpoint directories for model.bin files without config.json
2. Infers the model architecture and configuration from the state_dict
3. Generates appropriate config.json files

Usage:
    python generate_checkpoint_configs.py [--models-dir PATH] [--dry-run]
"""

import torch
import json
import os
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple


def detect_model_type(state_dict: Dict) -> Tuple[str, str]:
    """
    Detect model type and base architecture from state_dict keys.

    Returns:
        (model_type, base_architecture) tuple
    """
    keys = list(state_dict.keys())

    # Check for specific model architectures
    if any('encoder.decoder.' in key for key in keys):
        # Has decoder - T5ForConditionalGeneration
        return 'codet5_full', 't5'
    elif any('encoder.encoder.' in key for key in keys):
        # T5EncoderModel
        return 'codet5', 't5'
    elif any('roberta.' in key for key in keys):
        return 'roberta', 'roberta'
    elif any('bert.' in key for key in keys):
        return 'bert', 'bert'
    else:
        # Check for classifier to determine if it's a custom wrapper
        if 'classifier.weight' in keys:
            classifier_shape = state_dict['classifier.weight'].shape
            if classifier_shape[0] == 2:
                # 2-class classifier - likely NatGen or CodeT5Full
                return 'natgen', 't5'

        return 'unknown', 'unknown'


def infer_num_labels(state_dict: Dict) -> int:
    """Infer num_labels from classifier head shape."""
    if 'classifier.weight' in state_dict:
        return state_dict['classifier.weight'].shape[0]
    elif 'encoder.classifier.weight' in state_dict:
        return state_dict['encoder.classifier.weight'].shape[0]
    else:
        # Default to 1 for binary classification with sigmoid
        return 1


def infer_hidden_size(state_dict: Dict, base_arch: str) -> int:
    """Infer hidden size from embeddings or other layers."""
    if base_arch == 'roberta':
        if 'roberta.embeddings.word_embeddings.weight' in state_dict:
            return state_dict['roberta.embeddings.word_embeddings.weight'].shape[1]
        elif 'encoder.embeddings.word_embeddings.weight' in state_dict:
            return state_dict['encoder.embeddings.word_embeddings.weight'].shape[1]
    elif base_arch == 't5':
        if 'encoder.shared.weight' in state_dict:
            return state_dict['encoder.shared.weight'].shape[1]
        elif 'shared.weight' in state_dict:
            return state_dict['shared.weight'].shape[1]
        # For wrapped T5 models
        if 'classifier.weight' in state_dict:
            return state_dict['classifier.weight'].shape[1]
    elif base_arch == 'bert':
        if 'bert.embeddings.word_embeddings.weight' in state_dict:
            return state_dict['bert.embeddings.word_embeddings.weight'].shape[1]

    # Default for base models
    return 768


def create_roberta_config(state_dict: Dict, num_labels: int) -> Dict:
    """Create config for RoBERTa-based models."""
    hidden_size = infer_hidden_size(state_dict, 'roberta')

    # Try to get vocab size
    vocab_size = 50265  # default for RoBERTa
    if 'roberta.embeddings.word_embeddings.weight' in state_dict:
        vocab_size = state_dict['roberta.embeddings.word_embeddings.weight'].shape[0]
    elif 'encoder.embeddings.word_embeddings.weight' in state_dict:
        vocab_size = state_dict['encoder.embeddings.word_embeddings.weight'].shape[0]

    # Count layers
    layer_keys = [k for k in state_dict.keys() if 'encoder.layer.' in k]
    if layer_keys:
        layer_numbers = [int(k.split('encoder.layer.')[1].split('.')[0]) for k in layer_keys]
        num_hidden_layers = max(layer_numbers) + 1
    else:
        num_hidden_layers = 12  # default

    return {
        "_name_or_path": "microsoft/codebert-base",
        "architectures": ["RobertaForSequenceClassification"],
        "attention_probs_dropout_prob": 0.1,
        "bos_token_id": 0,
        "eos_token_id": 2,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": hidden_size,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-05,
        "max_position_embeddings": 514,
        "model_type": "roberta",
        "num_attention_heads": 12,
        "num_hidden_layers": num_hidden_layers,
        "num_labels": num_labels,
        "pad_token_id": 1,
        "position_embedding_type": "absolute",
        "type_vocab_size": 1,
        "vocab_size": vocab_size
    }


def create_t5_config(state_dict: Dict, num_labels: int, is_full: bool = False) -> Dict:
    """Create config for T5-based models (CodeT5)."""
    hidden_size = infer_hidden_size(state_dict, 't5')

    config = {
        "_name_or_path": "Salesforce/codet5-base",
        "architectures": ["T5ForConditionalGeneration" if is_full else "T5EncoderModel"],
        "d_ff": 2048,
        "d_kv": 64,
        "d_model": hidden_size,
        "decoder_start_token_id": 0,
        "dropout_rate": 0.1,
        "eos_token_id": 1,
        "feed_forward_proj": "relu",
        "initializer_factor": 1.0,
        "is_encoder_decoder": is_full,
        "layer_norm_epsilon": 1e-06,
        "model_type": "t5",
        "num_decoder_layers": 12 if is_full else None,
        "num_heads": 12,
        "num_layers": 12,
        "num_labels": num_labels,
        "output_past": True,
        "pad_token_id": 0,
        "relative_attention_num_buckets": 32,
        "tie_word_embeddings": False,
        "vocab_size": 32100
    }

    # Remove None values
    return {k: v for k, v in config.items() if v is not None}


def create_config_for_checkpoint(checkpoint_path: str) -> Optional[Dict]:
    """
    Load checkpoint and create appropriate config.

    Args:
        checkpoint_path: Path to model.bin file

    Returns:
        Config dict or None if unable to create
    """
    try:
        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu')

        # Detect model type
        model_type, base_arch = detect_model_type(state_dict)
        num_labels = infer_num_labels(state_dict)

        print(f"  Detected: model_type={model_type}, base_arch={base_arch}, num_labels={num_labels}")

        # Create appropriate config
        if base_arch == 'roberta':
            config = create_roberta_config(state_dict, num_labels)
        elif base_arch == 't5':
            is_full = model_type in ['codet5_full', 'natgen']
            config = create_t5_config(state_dict, num_labels, is_full)
        else:
            print(f"  Warning: Unknown base architecture '{base_arch}', skipping")
            return None

        return config

    except Exception as e:
        print(f"  Error processing {checkpoint_path}: {e}")
        return None


def scan_and_generate_configs(models_dir: str, dry_run: bool = False):
    """
    Scan models directory for checkpoints without config.json and generate them.

    Args:
        models_dir: Root directory containing model checkpoints
        dry_run: If True, only print what would be done without creating files
    """
    models_path = Path(models_dir)

    if not models_path.exists():
        print(f"Error: Models directory not found: {models_dir}")
        return

    # Find all checkpoint-best-acc directories with model.bin but no config.json
    checkpoints_found = 0
    configs_generated = 0

    for model_bin in models_path.rglob("checkpoint-best-acc/model.bin"):
        checkpoints_found += 1
        checkpoint_dir = model_bin.parent
        config_path = checkpoint_dir / "config.json"

        if config_path.exists():
            print(f"Skipping {checkpoint_dir} (config.json already exists)")
            continue

        print(f"\nProcessing: {checkpoint_dir}")

        # Generate config
        config = create_config_for_checkpoint(str(model_bin))

        if config is None:
            continue

        if dry_run:
            print(f"  [DRY RUN] Would create: {config_path}")
            print(f"  Config preview: num_labels={config.get('num_labels')}, "
                  f"model_type={config.get('model_type')}, "
                  f"architectures={config.get('architectures')}")
        else:
            # Save config
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"  ✓ Created: {config_path}")
            configs_generated += 1

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Checkpoints found: {checkpoints_found}")
    print(f"  Configs generated: {configs_generated}")
    if dry_run:
        print(f"  (DRY RUN - no files were created)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate config.json files for existing model checkpoints"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="/projects/sciences/computing/norja159/research/models",
        help="Root directory containing model checkpoints"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without creating files"
    )

    args = parser.parse_args()

    print(f"Scanning models directory: {args.models_dir}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'GENERATING CONFIGS'}")
    print(f"{'='*60}\n")

    scan_and_generate_configs(args.models_dir, args.dry_run)


if __name__ == "__main__":
    main()

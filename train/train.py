"""CLI entry point for SWE-Pruner training."""

import argparse
import sys

from .config import TrainConfig
from .trainer import train


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train SWE-Pruner model")

    # Model
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--backbone-model-name", type=str, default="Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--compression-head-type", type=str, default="crf")
    parser.add_argument("--bottleneck", type=int, default=256)
    parser.add_argument("--num-unfrozen-backbone-layers", type=int, default=2)

    # Data
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--val-data", type=str, default="")
    parser.add_argument("--max-seq-length", type=int, default=8192)

    # Optimization
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per-device-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lambda-rerank", type=float, default=0.05)

    # Logging
    parser.add_argument("--output-dir", type=str, default="./checkpoints")
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)

    # Wandb
    parser.add_argument("--wandb-project", type=str, default="swe-pruner")
    parser.add_argument("--wandb-entity", type=str, default="morphllm")
    parser.add_argument("--wandb-run-name", type=str, default="")

    args = parser.parse_args()

    config = TrainConfig(
        model_path=args.model_path,
        backbone_model_name=args.backbone_model_name,
        compression_head_type=args.compression_head_type,
        bottleneck=args.bottleneck,
        num_unfrozen_backbone_layers=args.num_unfrozen_backbone_layers,
        train_data=args.train_data,
        val_data=args.val_data,
        max_seq_length=args.max_seq_length,
        epochs=args.epochs,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        lambda_rerank=args.lambda_rerank,
        output_dir=args.output_dir,
        eval_steps=args.eval_steps,
        log_steps=args.log_steps,
        seed=args.seed,
        num_workers=args.num_workers,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
    )

    return config


def main():
    config = parse_args()
    train(config)


if __name__ == "__main__":
    main()

"""Training configuration for SWE-Pruner."""

from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    # Model
    model_path: str = ""  # Existing checkpoint OR empty to build from backbone
    backbone_model_name: str = "Qwen/Qwen3-Reranker-0.6B"
    compression_head_type: str = "crf"
    use_multi_layer_fusion: bool = True
    bottleneck: int = 256
    num_fusion_layers: int = 1
    num_heads: int = 8
    dropout: float = 0.4

    # Freezing
    num_unfrozen_backbone_layers: int = 2  # Unfreeze last N layers of backbone

    # Data
    train_data: str = ""
    val_data: str = ""
    max_seq_length: int = 8192

    # Optimization (from paper Section D.1)
    epochs: int = 3
    per_device_batch_size: int = 16
    gradient_accumulation_steps: int = 8  # effective batch = 16 * 8 = 128 on 1 GPU
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    lambda_rerank: float = 0.05  # Loss balance: 0.95*CRF + 0.05*MSE

    # Logging/checkpointing
    output_dir: str = "./checkpoints"
    eval_steps: int = 200
    log_steps: int = 10
    seed: int = 42
    num_workers: int = 4

    # Wandb
    wandb_project: str = "swe-pruner"
    wandb_entity: str = "morphllm"
    wandb_run_name: str = ""

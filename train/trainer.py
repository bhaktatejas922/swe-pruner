"""Pure PyTorch training loop for SWE-Pruner."""

import os
import sys
import time
import json
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer

try:
    import wandb
except ImportError:
    wandb = None

_SWE_PRUNER_SRC = os.path.join(
    os.path.dirname(__file__), "..", "swe-pruner", "src"
)
if _SWE_PRUNER_SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(_SWE_PRUNER_SRC))

from swe_pruner.swepruner import SwePrunerForCodeCompression
from swe_pruner.configuration import SwePrunerConfig

from .config import TrainConfig
from .dataset import SwePrunerDataset, collate_fn
from .loss import SwePrunerLoss
from .evaluate import evaluate

logger = logging.getLogger(__name__)


def build_model(config: TrainConfig, device: torch.device) -> SwePrunerForCodeCompression:
    """Build or load the model."""
    if config.model_path and os.path.isdir(config.model_path):
        logger.info(f"Loading model from {config.model_path}")
        model = SwePrunerForCodeCompression.from_pretrained(
            config.model_path,
            device_map=None,
            low_cpu_mem_usage=False,
        )
    else:
        logger.info(f"Building model from backbone {config.backbone_model_name}")
        swe_config = SwePrunerConfig(
            backbone_model_name_or_path=config.backbone_model_name,
            compression_head_type=config.compression_head_type,
            use_multi_layer_fusion=config.use_multi_layer_fusion,
            bottleneck=config.bottleneck,
            num_fusion_layers=config.num_fusion_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
            torch_dtype="bfloat16",
        )
        model = SwePrunerForCodeCompression(swe_config)

    # Move to device. Backbone in bfloat16 for flash attention, heads stay float32.
    model = model.to(device)
    model.model.backbone.to(dtype=torch.bfloat16)
    # Ensure fusion layers and heads are float32 (from_pretrained may cast all to bf16)
    model.model.fusion_layers.to(dtype=torch.float32)
    model.model.fusion_norms.to(dtype=torch.float32)
    model.model.compression_head.to(dtype=torch.float32)
    return model


def freeze_backbone(model: SwePrunerForCodeCompression, num_unfrozen: int = 2):
    """Freeze backbone except last N layers + final norm."""
    scorer = model.model  # TokenScorer
    backbone = scorer.backbone

    # Freeze ALL backbone params
    for p in backbone.parameters():
        p.requires_grad = False

    # Unfreeze last N layers
    num_layers = backbone.config.num_hidden_layers
    for i in range(num_layers - num_unfrozen, num_layers):
        for p in backbone.layers[i].parameters():
            p.requires_grad = True

    # Unfreeze final norm
    if hasattr(backbone, "norm"):
        for p in backbone.norm.parameters():
            p.requires_grad = True

    # fusion_layers, fusion_norms, compression_head are NOT part of backbone
    # They are already trainable (requires_grad=True by default)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Frozen backbone. Total params: {total:,}, Trainable: {trainable:,} "
        f"({100 * trainable / total:.1f}%)"
    )


def training_forward(model, input_ids, attention_mask):
    """Like model.forward() but returns raw CRF emissions instead of collapsed logits.

    Returns:
        emissions: [B, L, 2] raw CRF emission scores.
        score_logits: [B] log P(yes) from reranking head.
    """
    scorer = model.model  # TokenScorer

    # Backbone is already in bfloat16 for flash attention compatibility
    backbone_out = scorer.backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )

    # Multi-layer fusion (in fp32, matching TokenScorer.forward)
    if scorer.use_multi_layer_fusion:
        hs = backbone_out.hidden_states
        fused = torch.cat([
            hs[scorer.early_layer_idx].float(),
            hs[scorer.middle_layer_idx].float(),
            hs[scorer.final_layer_idx].float(),
        ], dim=-1)
    else:
        fused = backbone_out.hidden_states[-1].float()

    # Fusion attention
    h = fused
    kpm = (attention_mask == 0).to(h.device)
    for attn, norm in zip(scorer.fusion_layers, scorer.fusion_norms):
        out, _ = attn(h, h, h, key_padding_mask=kpm)
        h = norm(out + h)
    h = scorer.dropout(h)

    # CRF emissions (raw, not collapsed)
    emissions = scorer.compression_head.feature_extractor(h)  # [B, L, 2]

    # Reranking score (same as TokenScorer.forward)
    last_hidden = backbone_out.last_hidden_state.float()
    B = last_hidden.size(0)
    last_idx = attention_mask.sum(dim=1) - 1
    last_idx = torch.clamp(last_idx, min=0)
    cls_h = last_hidden[torch.arange(B, device=last_hidden.device), last_idx]
    vocab_logits = cls_h @ scorer.embedding_layer.weight.float().T
    no_l = vocab_logits[:, scorer.token_no_id]
    yes_l = vocab_logits[:, scorer.token_yes_id]
    score_logits = F.log_softmax(torch.stack([no_l, yes_l], dim=1), dim=1)[:, 1]

    return emissions, score_logits


def get_param_groups(model: SwePrunerForCodeCompression, config: TrainConfig):
    """Create parameter groups with differential learning rates."""
    scorer = model.model
    backbone = scorer.backbone

    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Backbone layers get lower LR
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    return [
        {"params": backbone_params, "lr": config.learning_rate * 0.1, "weight_decay": config.weight_decay},
        {"params": head_params, "lr": config.learning_rate, "weight_decay": config.weight_decay},
    ]


def train(config: TrainConfig):
    """Main training loop."""
    # Setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Wandb
    use_wandb = wandb is not None and config.wandb_project
    if use_wandb:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity or None,
            name=config.wandb_run_name or None,
            config={
                "model_path": config.model_path,
                "backbone": config.backbone_model_name,
                "compression_head": config.compression_head_type,
                "multi_layer_fusion": config.use_multi_layer_fusion,
                "bottleneck": config.bottleneck,
                "num_fusion_layers": config.num_fusion_layers,
                "dropout": config.dropout,
                "unfrozen_layers": config.num_unfrozen_backbone_layers,
                "max_seq_length": config.max_seq_length,
                "epochs": config.epochs,
                "per_device_batch_size": config.per_device_batch_size,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "effective_batch_size": config.per_device_batch_size * config.gradient_accumulation_steps,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "warmup_ratio": config.warmup_ratio,
                "lambda_rerank": config.lambda_rerank,
                "seed": config.seed,
            },
        )
        logger.info(f"Wandb run: {wandb.run.url}")

    # Build model
    model = build_model(config, device)
    freeze_backbone(model, config.num_unfrozen_backbone_layers)

    # Tokenizer (reuse from model)
    tokenizer = model.tokenizer

    # Datasets
    train_dataset = SwePrunerDataset(
        config.train_data, tokenizer, config.max_seq_length
    )
    logger.info(f"Train samples: {len(train_dataset)}")

    val_dataset = None
    val_loader = None
    if config.val_data and os.path.exists(config.val_data):
        val_dataset = SwePrunerDataset(
            config.val_data, tokenizer, config.max_seq_length
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.per_device_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=config.num_workers,
            pin_memory=True,
        )
        logger.info(f"Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.per_device_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Loss
    loss_fn = SwePrunerLoss(lambda_rerank=config.lambda_rerank)

    # Optimizer
    param_groups = get_param_groups(model, config)
    optimizer = AdamW(param_groups)

    # Scheduler
    steps_per_epoch = len(train_loader) // config.gradient_accumulation_steps
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    logger.info(
        f"Steps/epoch: {steps_per_epoch}, Total steps: {total_steps}, "
        f"Warmup: {warmup_steps}"
    )

    # Training state
    os.makedirs(config.output_dir, exist_ok=True)
    best_f1 = -1.0
    global_step = 0
    crf_layer = model.model.compression_head.crf

    model.train()
    optimizer.zero_grad()

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_tags = batch["token_tags"].to(device)
            code_mask = batch["code_mask"].to(device)
            doc_score = batch["doc_score"].to(device)

            # Forward
            emissions, score_logits = training_forward(
                model, input_ids, attention_mask
            )

            # Loss
            loss_dict = loss_fn(
                emissions, token_tags, code_mask,
                score_logits, doc_score, crf_layer,
            )
            loss = loss_dict["loss"] / config.gradient_accumulation_steps
            loss.backward()

            epoch_loss += loss_dict["loss"].item()
            epoch_steps += 1

            # Gradient accumulation step
            if (step + 1) % config.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    config.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % config.log_steps == 0:
                    avg_loss = epoch_loss / epoch_steps
                    lr = scheduler.get_last_lr()[0]
                    logger.info(
                        f"Epoch {epoch+1}/{config.epochs} "
                        f"Step {global_step}/{total_steps} "
                        f"Loss: {loss_dict['loss'].item():.4f} "
                        f"(CRF: {loss_dict['loss_compress'].item():.4f}, "
                        f"Rerank: {loss_dict['loss_rerank'].item():.4f}) "
                        f"LR: {lr:.2e}"
                    )
                    if use_wandb:
                        wandb.log({
                            "train/loss": loss_dict["loss"].item(),
                            "train/loss_crf": loss_dict["loss_compress"].item(),
                            "train/loss_rerank": loss_dict["loss_rerank"].item(),
                            "train/lr": lr,
                            "train/epoch": epoch + 1,
                        }, step=global_step)

                # Evaluation
                if val_loader and global_step % config.eval_steps == 0:
                    metrics = evaluate(
                        model, val_loader, training_forward, device
                    )
                    logger.info(
                        f"[Eval] Step {global_step} | "
                        f"F1: {metrics['line_f1']:.4f} | "
                        f"IoU: {metrics['line_iou']:.4f} | "
                        f"P: {metrics['line_precision']:.4f} | "
                        f"R: {metrics['line_recall']:.4f} | "
                        f"Score MSE: {metrics['doc_score_mse']:.4f} | "
                        f"Compression: {metrics['compression_ratio']:.4f}"
                    )

                    if use_wandb:
                        wandb.log({
                            "eval/line_f1": metrics["line_f1"],
                            "eval/line_iou": metrics["line_iou"],
                            "eval/line_precision": metrics["line_precision"],
                            "eval/line_recall": metrics["line_recall"],
                            "eval/doc_score_mse": metrics["doc_score_mse"],
                            "eval/compression_ratio": metrics["compression_ratio"],
                        }, step=global_step)

                    # Save best
                    if metrics["line_f1"] > best_f1:
                        best_f1 = metrics["line_f1"]
                        save_path = os.path.join(config.output_dir, "best")
                        model.save_pretrained(save_path)
                        tokenizer.save_pretrained(save_path)
                        logger.info(f"New best F1: {best_f1:.4f}, saved to {save_path}")

                    model.train()

        # End of epoch
        avg_loss = epoch_loss / max(epoch_steps, 1)
        logger.info(f"Epoch {epoch+1} done. Avg loss: {avg_loss:.4f}")

        # Save epoch checkpoint
        save_path = os.path.join(config.output_dir, f"epoch_{epoch+1}")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    # Final eval
    if val_loader:
        metrics = evaluate(model, val_loader, training_forward, device)
        logger.info(f"[Final Eval] {metrics}")

        # Save metrics
        with open(os.path.join(config.output_dir, "final_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    if use_wandb:
        wandb.finish()

    logger.info("Training complete.")
    return model

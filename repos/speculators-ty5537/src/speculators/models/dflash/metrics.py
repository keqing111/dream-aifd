"""Metrics and loss functions for DFlash draft model."""

from functools import partial
from typing import Any

import torch

from speculators.models.metrics import (
    LossConfig,
    compound_loss,
    compute_accuracy_multi_step,
    dflash_loss_decay,
    dpace_loss_decay,
    kl_div_loss,
)

_DEFAULT_LOSS_CONFIG: LossConfig = {"kl_div": (kl_div_loss, 1.0)}


def compute_metrics(
    logits: torch.Tensor,  # shape: [1, num_anchors*block_size, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, num_anchors*block_size, draft_vocab_size]
    loss_mask: torch.Tensor,  # shape: [1, num_anchors*block_size]
    block_size: int = 1,
    gamma: float = 4.0,
    loss_config: LossConfig | None = None,
    per_position_loss_weight: str = "fixed-exp-decay",
    dpace_alpha: float = 0.5,
    sample_from_anchor: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Compute loss and accuracy metrics for draft model predictions.

    Args:
        logits: Model logits [1, T, V]
        targets: Target logits [1, T, V]
        loss_mask: Binary mask [1, T]
        block_size: Block size for per-position metrics
        gamma: Temperature for exponential decay in loss weighting
        loss_config: Mapping of ``{name: (loss_fn, weight)}``
        per_position_loss_weight: Weighting option for per-position block-drafting loss
        dpace_alpha: Smoothing constant for D-Pace loss weighting

    Returns:
        Tuple of (loss, metrics_dict) where metrics_dict contains:
            - loss: Scalar loss value
            - full_acc: Overall accuracy
            - position {i} acc: Accuracy at position i within blocks
            - eal: Expected Accepted Length (headline speculative-decoding metric)
    """
    if loss_config is None:
        loss_config = _DEFAULT_LOSS_CONFIG
    seq_len = logits.shape[1]
    pos_idx = torch.arange(seq_len, device=logits.device) % block_size
    pos_idx = pos_idx.unsqueeze(0)  # shape: [1, T]

    if per_position_loss_weight == "dpace":
        decay_fn = partial(
            dpace_loss_decay,
            loss_mask=loss_mask,
            block_size=block_size,
            dpace_alpha=dpace_alpha,
        )
    else:
        decay_fn = partial(
            dflash_loss_decay, gamma=gamma, sample_from_anchor=sample_from_anchor
        )

    loss, term_losses = compound_loss(
        logits,
        targets,
        loss_mask,
        pos_idx,
        loss_config=loss_config,
        decay_fn=decay_fn,
    )

    pred_ids = torch.argmax(logits, dim=-1)
    target_ids = torch.argmax(targets, dim=-1)

    correct_per_pos, total_per_pos = compute_accuracy_multi_step(
        pred_ids, target_ids, loss_mask, pos_idx, block_size
    )

    ones = torch.tensor(1.0, device=logits.device)
    metrics: dict[str, Any] = {}
    metrics["loss_sum"] = loss.detach().clone()
    metrics["loss_total"] = ones
    for term_name, term_val in term_losses.items():
        metrics[f"{term_name}_sum"] = term_val
        metrics[f"{term_name}_total"] = ones.clone()

    # Start position: 0 if sample_from_anchor else 1 (skip anchor)
    start_pos = 0 if sample_from_anchor else 1
    metrics["full_acc_sum"] = correct_per_pos[start_pos:].sum()
    metrics["full_acc_total"] = total_per_pos[start_pos:].sum()

    # EAL = sum_k prod_{i<=k} acc_i over drafted positions
    eal = torch.zeros((), device=logits.device)
    cum = torch.ones((), device=logits.device)
    for pos in range(start_pos, block_size):
        metrics[f"position_{pos}_acc_sum"] = correct_per_pos[pos]
        metrics[f"position_{pos}_acc_total"] = total_per_pos[pos]
        acc = correct_per_pos[pos] / total_per_pos[pos].clamp(min=1.0)
        cum = cum * acc
        eal = eal + cum
    metrics["eal_sum"] = eal
    metrics["eal_total"] = ones.clone()
    return loss, metrics


def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """不带可学参数的逐 token RMSNorm。

    AIFD 的监督目标在不同样本上可能来自不同的层，层间尺度差实测 4.8x
    （候选窗口内），而且层 6 以上恒有一个 BOS token 的 norm 是中位数的 400 倍。
    两边各归一化一次，这两件事一起消掉。speculators 现有管线本来也在对隐状态
    做 RMSNorm（`verifier_norm` / `hidden_norm`），所以这不是新引入的处理。
    """
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)


def compute_aifd_loss(
    draft_hidden: torch.Tensor,  # [1, T, hidden_size]，norm 之后、lm_head 之前
    aifd_hidden_states: torch.Tensor,  # [1, T, hidden_size]，AIFD 选层的目标
    aligned_loss_mask: torch.Tensor,  # [1, num_anchors*block_size]
    anchored_block_indices: torch.Tensor,  # [num_anchors*block_size]
    norm: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """AIFD 中间层特征蒸馏：draft 的最终隐状态 → 熵选层的 target 隐状态。

    照搬 DREAM-S 的 `compute_mid_loss`（`train/main_deepspeed.py:244-247`）：
    `nn.SmoothL1Loss(reduction="none")` 的 masked mean。

    metric 走 `_sum`/`_total` 命名，让 `normalize_counted_metrics` 折算成
    `aifd_loss`（= 每个有效位置的**平均** loss）。
    """
    pred = draft_hidden[:, anchored_block_indices]  # [1, A*bs, H]
    tgt = aifd_hidden_states[:, anchored_block_indices]  # [1, A*bs, H]
    if norm:
        pred = _rms_norm(pred)
        tgt = _rms_norm(tgt)

    m = aligned_loss_mask.unsqueeze(-1).to(torch.float32)  # [1, A*bs, 1]
    per_pos = torch.nn.functional.smooth_l1_loss(
        pred.float(), tgt.float(), reduction="none"
    ).mean(-1)  # [1, A*bs]

    aifd_sum = (per_pos * m.squeeze(-1)).sum()
    aifd_total = m.sum().clamp_min(1.0)
    loss = aifd_sum / aifd_total
    metrics = {
        "aifd_loss_sum": aifd_sum.detach(),
        "aifd_loss_total": aifd_total.detach(),
    }
    return loss, metrics

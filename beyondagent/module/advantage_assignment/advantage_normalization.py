"""
Advantage Normalization Module
优势归一化模块 - 支持三种归一化策略
"""

import torch
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AdvNormConfig:
    """优势归一化配置"""
    enable: bool = True
    level: str = "batch"  # "batch" | "group"
    group_size: Optional[int] = None
    normalization_type: str = "with_std"  # "with_std" | "no_std" | "batch_std"


def normalize_advantages(
    advantages: torch.Tensor,
    mask: torch.Tensor,
    config: AdvNormConfig,
    rollout_n: int = 8
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    对优势进行归一化处理
    
    Args:
        advantages: 优势张量 (bs, seq_len)
        mask: 有效token的mask (bs, seq_len)
        config: 归一化配置
        rollout_n: rollout重复次数，用于group模式的默认group_size
    
    Returns:
        normalized_advantages: 归一化后的优势张量
        stats: 统计信息字典
    """
    if not config.enable:
        return advantages, {}
    
    device = advantages.device
    bs, seq_len = advantages.shape
    
    # 只对非零且有效的token进行归一化
    nonzero_mask = mask.bool() & (advantages != 0)
    norm_adv = advantages.clone()
    
    # 统计信息
    stats = {
        "level": config.level,
        "normalization_type": config.normalization_type,
        "tokens_normed": 0,
        "groups": 0,
        "zero_groups": 0,
        "median_mean": 0.0,
        "std_mean": 0.0,
    }
    
    if config.level == "batch":
        norm_adv, batch_stats = _normalize_batch_level(
            advantages, norm_adv, nonzero_mask, config.normalization_type
        )
        stats.update(batch_stats)
        
    elif config.level == "group":
        group_size = config.group_size or rollout_n
        norm_adv, group_stats = _normalize_group_level(
            advantages, norm_adv, nonzero_mask, group_size, config.normalization_type, device
        )
        stats.update(group_stats)
        
    else:
        raise ValueError(f"Unknown level: {config.level}")
    
    # 计算归一化后的统计信息
    final_stats = _compute_final_stats(norm_adv, mask)
    stats.update(final_stats)
    
    return norm_adv, stats


def _normalize_batch_level(
    advantages: torch.Tensor,
    norm_adv: torch.Tensor,
    nonzero_mask: torch.Tensor,
    normalization_type: str
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """批次级别归一化"""
    nz_adv = advantages[nonzero_mask]
    
    if nz_adv.numel() > 0:
        med = torch.median(nz_adv)
        std = nz_adv.std(unbiased=False).clamp_min(1e-8)
        
        # 应用归一化
        if normalization_type == "with_std":
            norm_adv[nonzero_mask] = (advantages[nonzero_mask] - med) / std
        elif normalization_type == "no_std":
            norm_adv[nonzero_mask] = advantages[nonzero_mask] - med
        elif normalization_type == "batch_std":
            # 在batch模式下，batch_std等同于with_std
            norm_adv[nonzero_mask] = (advantages[nonzero_mask] - med) / std
        else:
            raise ValueError(f"Unknown normalization_type: {normalization_type}")
    else:
        med = torch.tensor(0.0, device=advantages.device)
        std = torch.tensor(1.0, device=advantages.device)
    
    return norm_adv, {
        "groups": 1,
        "tokens_normed": int(nonzero_mask.sum().item()),
        "median_mean": float(med),
        "std_mean": float(std),
        "zero_groups": 0,
    }


def _normalize_group_level(
    advantages: torch.Tensor,
    norm_adv: torch.Tensor,
    nonzero_mask: torch.Tensor,
    group_size: int,
    normalization_type: str,
    device: torch.device
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """组级别归一化"""
    bs = advantages.shape[0]
    group_ids = torch.arange(bs, device=device) // group_size
    
    tokens_normed = 0
    med_list, std_list = [], []
    zero_groups = 0
    
    # 如果是batch_std模式，预先计算全局统计量
    global_std = None
    if normalization_type == "batch_std":
        nz_adv_global = advantages[nonzero_mask]
        if nz_adv_global.numel() > 0:
            global_std = nz_adv_global.std(unbiased=False).clamp_min(1e-8)
    
    # 按组处理
    for gid in group_ids.unique():
        g_sample_mask = (group_ids == gid).unsqueeze(1)
        g_mask = g_sample_mask & nonzero_mask
        
        if not g_mask.any():
            continue
        
        g_adv = advantages[g_mask]
        med = torch.median(g_adv)
        std = g_adv.std(unbiased=False)
        
        # 应用归一化
        if normalization_type == "with_std":
            # 使用组内std
            if std <= 1e-8:
                zero_groups += 1
                continue
            norm_adv[g_mask] = (advantages[g_mask] - med) / std
            
        elif normalization_type == "no_std":
            # 只减组内中位数
            norm_adv[g_mask] = advantages[g_mask] - med
            
        elif normalization_type == "batch_std":
            # 使用全局std，组内median
            if global_std is None or global_std <= 1e-8:
                norm_adv[g_mask] = advantages[g_mask] - med  # fallback
            else:
                norm_adv[g_mask] = (advantages[g_mask] - med) / global_std
        else:
            raise ValueError(f"Unknown normalization_type: {normalization_type}")
        
        med_list.append(med)
        std_list.append(std)
        tokens_normed += int(g_adv.numel())
    
    return norm_adv, {
        "groups": int(group_ids.unique().numel()),
        "tokens_normed": tokens_normed,
        "median_mean": torch.stack(med_list).mean().item() if med_list else 0.0,
        "std_mean": torch.stack(std_list).mean().item() if std_list else 1.0,
        "zero_groups": zero_groups,
    }


def _compute_final_stats(norm_adv: torch.Tensor, mask: torch.Tensor) -> Dict[str, Any]:
    """计算归一化后的最终统计信息"""
    mask_bool = mask.bool()
    
    pos_tokens = int((norm_adv[mask_bool] > 0).sum().item())
    neg_tokens = int((norm_adv[mask_bool] < 0).sum().item())
    zero_tokens = int((norm_adv[mask_bool] == 0).sum().item())
    
    seq_sum = (norm_adv * mask_bool).sum(dim=1)
    pos_sequences = int((seq_sum > 0).sum().item())
    neg_sequences = int((seq_sum < 0).sum().item())
    zero_sequences = int((seq_sum == 0).sum().item())
    
    return {
        "pos_tokens": pos_tokens,
        "neg_tokens": neg_tokens,
        "zero_tokens": zero_tokens,
        "pos_sequences": pos_sequences,
        "neg_sequences": neg_sequences,
        "zero_sequences": zero_sequences,
        "neg_token_ratio": neg_tokens / max(1, pos_tokens + neg_tokens),
    }


def extract_adv_norm_config(config) -> AdvNormConfig:
    """从主配置中提取advantage normalization配置"""
    norm_root = getattr(config, "semantic_advantage", None)
    adv_norm_cfg = getattr(norm_root, "adv_norm", None) if norm_root else None
    
    if not adv_norm_cfg:
        return AdvNormConfig(enable=False)
    
    return AdvNormConfig(
        enable=getattr(adv_norm_cfg, "enable", True),
        level=getattr(adv_norm_cfg, "level", "batch"),
        group_size=getattr(adv_norm_cfg, "group_size", None),
        normalization_type=getattr(adv_norm_cfg, "normalization_type", "with_std"),
    )
from __future__ import annotations

import os
import copy
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def fedavg_aggregate(state_dicts: List[Dict[str, torch.Tensor]], weights: List[int]) -> Dict[str, torch.Tensor]:
    """
    FedAvg: weighted average by number of local train samples.
    修复：正确处理整数类型参数（如位置编码的索引），只聚合浮点类型参数。
    """
    total = float(sum(weights)) if sum(weights) > 0 else 1.0
    
    # 获取参考 state_dict
    ref_sd = state_dicts[0]
    out = {}
    
    for key in ref_sd.keys():
        # 检查参数类型
        param_dtype = ref_sd[key].dtype
        
        # 只聚合浮点类型参数（可训练参数）
        if param_dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
            # 初始化聚合结果（使用 float32 进行计算）
            aggregated = torch.zeros_like(ref_sd[key], dtype=torch.float32)
            
            for sd, w in zip(state_dicts, weights):
                alpha = float(w) / total
                aggregated += sd[key].float() * alpha
            
            # 转回原始类型（如果需要）
            if param_dtype != torch.float32:
                out[key] = aggregated.to(param_dtype)
            else:
                out[key] = aggregated
        else:
            # 整数/布尔类型参数（通常是 buffer 或不训练的参数）：直接复制第一个客户端的
            # 这些参数在 FedAvg 中不应该被平均
            out[key] = ref_sd[key].clone()
    
    return out


def clip_client_update(
    state_dict: Dict[str, torch.Tensor],
    reference_state_dict: Dict[str, torch.Tensor],
    max_update_norm: float,
) -> Dict[str, torch.Tensor]:
    """Build the clipped model-delta payload uploaded by one DP-FedAvg client.

    The client subtracts the model received at the start of the aggregation
    from its locally trained model and clips the complete floating-point update
    to one global L2 norm.  Only this clipped update is returned; absolute model
    parameters and non-floating buffers are not part of the upload.
    """
    clip_norm = float(max_update_norm)
    if not np.isfinite(clip_norm) or clip_norm <= 0.0:
        raise ValueError("max_update_norm must be finite and greater than zero.")

    reference_keys = tuple(reference_state_dict.keys())
    if set(state_dict.keys()) != set(reference_keys):
        raise ValueError("state_dict keys do not match reference_state_dict.")

    updates: Dict[str, torch.Tensor] = {}
    for key in reference_keys:
        local_value = state_dict[key]
        reference_value = reference_state_dict[key]
        if local_value.shape != reference_value.shape:
            raise ValueError(
                f"Tensor {key!r} has shape {tuple(local_value.shape)}, expected "
                f"{tuple(reference_value.shape)}."
            )
        if reference_value.is_floating_point():
            updates[key] = (
                local_value.detach().to(device="cpu", dtype=torch.float32)
                - reference_value.detach().to(device="cpu", dtype=torch.float32)
            )

    if not updates:
        raise ValueError("reference_state_dict has no floating-point tensors to clip.")
    if any(not bool(torch.all(torch.isfinite(update))) for update in updates.values()):
        raise ValueError("Client update contains NaN or infinite values.")

    squared_norm = sum(torch.sum(update * update) for update in updates.values())
    update_norm = torch.sqrt(squared_norm)
    clip_factor = torch.clamp(
        torch.as_tensor(clip_norm, dtype=update_norm.dtype)
        / (update_norm + torch.finfo(update_norm.dtype).eps),
        max=1.0,
    )
    return {key: update.mul(clip_factor) for key, update in updates.items()}


def dp_fedavg_aggregate(
    clipped_updates: List[Dict[str, torch.Tensor]],
    weights: List[int],
    reference_state_dict: Dict[str, torch.Tensor],
    max_update_norm: float,
    noise_multiplier: float,
    *,
    adjacency: str = "add_remove",
    generator: Optional[torch.Generator] = None,
    sensitivity_multiplier: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Aggregate already-clipped client uploads and add central Gaussian noise.

    Clients must create every payload with :func:`clip_client_update` before
    transmission.  This server-side function never receives an absolute local
    model and does not perform clipping.  It validates the norm bound, averages
    the clipped updates, and adds Gaussian noise.  Non-floating state entries
    are copied from ``reference_state_dict``.

    ``weights`` are public. The legacy path treats participants as public. For
    ``add_remove`` adjacency the noise standard deviation is
    ``noise_multiplier * max_update_norm * max(weights) / sum(weights)``.
    ``replace_one`` uses twice that sensitivity. Privacy accounting across
    rounds and private client sampling must be handled by the caller. The
    amplified fixed-size add/remove FHF path uses uniform weights, hides
    participation in the model release, and sets ``sensitivity_multiplier=2``
    for the 2C/m bound in Birrell et al. (2024), Theorem 3.1.

    Args:
        clipped_updates: Client-side clipped model-delta payloads.
        weights: Public, non-negative client aggregation weights.
        reference_state_dict: The model sent to clients at the start of the
            round. Clipping full model states instead of updates is incorrect.
        max_update_norm: Maximum global L2 norm of each client update.
        noise_multiplier: Gaussian noise standard deviation divided by the
            selected adjacency sensitivity. Zero is useful for testing but
            does not provide differential privacy.
        adjacency: Either ``"add_remove"`` or ``"replace_one"``.
        generator: Optional PyTorch random generator.
    """
    if not clipped_updates:
        raise ValueError("clipped_updates must contain at least one client upload.")
    if len(clipped_updates) != len(weights):
        raise ValueError(
            f"clipped_updates and weights must have the same length, got "
            f"{len(clipped_updates)} and {len(weights)}."
        )

    clip_norm = float(max_update_norm)
    if not np.isfinite(clip_norm) or clip_norm <= 0.0:
        raise ValueError("max_update_norm must be finite and greater than zero.")
    noise = float(noise_multiplier)
    if not np.isfinite(noise) or noise < 0.0:
        raise ValueError("noise_multiplier must be finite and non-negative.")

    weight_array = np.asarray(weights, dtype=np.float64)
    if weight_array.ndim != 1 or not np.all(np.isfinite(weight_array)):
        raise ValueError("weights must be a one-dimensional list of finite values.")
    if np.any(weight_array < 0.0):
        raise ValueError("weights must be non-negative.")
    total_weight = float(weight_array.sum())
    if total_weight <= 0.0:
        raise ValueError("At least one aggregation weight must be greater than zero.")

    adjacency_name = str(adjacency).strip().lower()
    if adjacency_name not in {"add_remove", "replace_one"}:
        raise ValueError("adjacency must be 'add_remove' or 'replace_one'.")

    float_keys = [
        key
        for key, value in reference_state_dict.items()
        if value.is_floating_point()
    ]
    if not float_keys:
        raise ValueError("reference_state_dict has no floating-point tensors to aggregate.")

    expected_keys = set(float_keys)
    norm_tolerance = max(1e-7, clip_norm * 1e-5)
    for client_index, client_update in enumerate(clipped_updates):
        if set(client_update.keys()) != expected_keys:
            raise ValueError(
                f"Client {client_index} upload must contain exactly the "
                "floating-point update tensors."
            )
        for key in float_keys:
            if client_update[key].shape != reference_state_dict[key].shape:
                raise ValueError(
                    f"Client {client_index} update {key!r} has shape "
                    f"{tuple(client_update[key].shape)}, expected "
                    f"{tuple(reference_state_dict[key].shape)}."
                )
            if not bool(torch.all(torch.isfinite(client_update[key]))):
                raise ValueError(
                    f"Client {client_index} update {key!r} contains NaN or infinity."
                )
        squared_norm = sum(
            torch.sum(client_update[key].detach().float() ** 2)
            for key in float_keys
        )
        if float(torch.sqrt(squared_norm)) > clip_norm + norm_tolerance:
            raise ValueError(
                f"Client {client_index} upload exceeds max_update_norm; "
                "client-side clipping is required before aggregation."
            )

    if not np.isfinite(sensitivity_multiplier) or sensitivity_multiplier < 1.0:
        raise ValueError("sensitivity_multiplier must be finite and at least one.")
    sensitivity_factor = (2.0 if adjacency_name == "replace_one" else 1.0) * sensitivity_multiplier
    sensitivity = (
        sensitivity_factor
        * clip_norm
        * float(weight_array.max())
        / total_weight
    )
    noise_std = noise * sensitivity

    out: Dict[str, torch.Tensor] = {}
    for key, reference_value in reference_state_dict.items():
        if not reference_value.is_floating_point():
            out[key] = reference_value.detach().clone()
            continue

        aggregated_update = torch.zeros_like(reference_value, dtype=torch.float32)
        for client_update, weight in zip(clipped_updates, weight_array):
            aggregated_update.add_(
                client_update[key].to(
                    device=aggregated_update.device,
                    dtype=aggregated_update.dtype,
                ),
                alpha=float(weight) / total_weight,
            )
        if noise_std > 0.0:
            aggregated_update.add_(
                torch.randn(
                    aggregated_update.shape,
                    dtype=aggregated_update.dtype,
                    device=aggregated_update.device,
                    generator=generator,
                ),
                alpha=noise_std,
            )
        out[key] = (
            reference_value.detach().to(dtype=torch.float32) + aggregated_update
        ).to(dtype=reference_value.dtype)

    return out

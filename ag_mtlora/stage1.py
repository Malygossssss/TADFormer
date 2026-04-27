import csv
import json
import os
import random
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from yacs.config import CfgNode as CN

from ag_mtlora.config_utils import (
    build_task_to_group,
    canonicalize_groups,
    enumerate_candidate_groups,
    enumerate_partitions,
    group_display_name,
    normalize_partition_granularity,
    resolve_group_shared_ranks,
)
from data.mtl_ds import (
    collate_mil,
    get_mtl_dataset,
    get_mtl_split_sample_ids,
    get_transformations,
)
from logger import create_logger
from models import build_model, build_mtl_model
from models.lora import mark_only_lora_as_trainable
from mtl_loss_schemes import MultiTaskLoss, get_loss
from optimizer import build_optimizer
from utils import load_checkpoint, load_pretrained, mkdir_if_missing


SEARCH_SCORE_SOURCE_GROUP_PROXY = "group_proxy"
DEFAULT_TASK_LOSS_WEIGHTS = {
    "depth": 1.0,
    "semseg": 1.0,
    "human_parts": 2.0,
    "sal": 5.0,
    "edge": 50.0,
    "normals": 10.0,
}


def set_random_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_stage1_logger(output_root: str):
    mkdir_if_missing(output_root)
    return create_logger(output_dir=output_root, dist_rank=0, name="ag_mtlora_stage1")


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(payload: Dict, path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        mkdir_if_missing(directory)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)


def save_rows_csv(rows, path: str, fieldnames=None) -> None:
    directory = os.path.dirname(path)
    if directory:
        mkdir_if_missing(directory)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_matrix_csv(tasks: Sequence[str], matrix: np.ndarray, path: str) -> None:
    rows = []
    for src_idx, src_task in enumerate(tasks):
        row = {"source_task": src_task}
        for dst_idx, dst_task in enumerate(tasks):
            row[dst_task] = float(matrix[src_idx, dst_idx])
        rows.append(row)
    save_rows_csv(rows, path, fieldnames=["source_task", *list(tasks)])


def group_to_key(group: Sequence[str]) -> str:
    return "+".join(str(task) for task in group)


def normalize_search_score_source(search_score_source: str) -> str:
    source = str(search_score_source or SEARCH_SCORE_SOURCE_GROUP_PROXY)
    if source != SEARCH_SCORE_SOURCE_GROUP_PROXY:
        raise ValueError("This TADFormer port supports only SEARCH_SCORE_SOURCE='group_proxy'.")
    return source


def get_search_score_source(config: CN) -> str:
    return normalize_search_score_source(config.MODEL.AGMTLORA.SEARCH_SCORE_SOURCE)


def append_search_score_suffix(path: str, search_score_source: str) -> str:
    path = os.path.abspath(path)
    base, ext = os.path.splitext(path)
    if not ext:
        ext = ".json"
    return f"{base}__{normalize_search_score_source(search_score_source)}{ext}"


def _artifact_path(output_root: str, name: str, search_score_source: Optional[str] = None, ext: str = ".json") -> str:
    if search_score_source is not None:
        name = f"{name}__{normalize_search_score_source(search_score_source)}"
    return os.path.join(output_root, f"{name}{ext}")


def _get_stage1_db_name(config: CN) -> str:
    return str(config.DATA.DBNAME)


def build_stage1_data_split_manifest(config: CN, logger):
    mode = str(config.MODEL.AGMTLORA.DATA_SPLIT_MODE)
    if mode == "official_val":
        return {
            "mode": mode,
            "train_split": "train",
            "eval_split": "val",
            "meta_split_path": None,
        }
    if mode != "train_meta_strict":
        raise ValueError("MODEL.AGMTLORA.DATA_SPLIT_MODE must be 'train_meta_strict' or 'official_val'.")

    db_name = _get_stage1_db_name(config)
    sample_ids = get_mtl_split_sample_ids(db_name, config.DATA.DATA_PATH, split="train")
    if len(sample_ids) < 2:
        raise ValueError("Stage-1 train_meta_strict split requires at least 2 train samples.")

    indices = list(range(len(sample_ids)))
    rng = random.Random(int(config.MODEL.AGMTLORA.RESOLVED_META_SPLIT_SEED))
    rng.shuffle(indices)
    val_count = int(round(len(indices) * float(config.MODEL.AGMTLORA.META_VAL_RATIO)))
    val_count = min(max(1, val_count), len(indices) - 1)
    meta_val_indices = sorted(indices[:val_count])
    meta_train_indices = sorted(indices[val_count:])

    manifest = {
        "mode": mode,
        "train_split": "train_meta_train",
        "eval_split": "train_meta_val",
        "seed": int(config.MODEL.AGMTLORA.RESOLVED_META_SPLIT_SEED),
        "meta_val_ratio": float(config.MODEL.AGMTLORA.META_VAL_RATIO),
        "all_train_ids": sample_ids,
        "meta_train_indices": meta_train_indices,
        "meta_val_indices": meta_val_indices,
        "meta_train_ids": [sample_ids[idx] for idx in meta_train_indices],
        "meta_val_ids": [sample_ids[idx] for idx in meta_val_indices],
        "meta_split_path": os.path.abspath(config.MODEL.AGMTLORA.META_SPLIT_SAVE_PATH),
    }
    save_json(manifest, manifest["meta_split_path"])
    logger.info(
        "Stage-1 meta split saved | train=%d | meta_val=%d | path=%s",
        len(meta_train_indices),
        len(meta_val_indices),
        manifest["meta_split_path"],
    )
    return manifest


def build_stage1_data_loaders(config: CN, data_split_manifest: Dict = None):
    db_name = _get_stage1_db_name(config)
    train_transforms, val_transforms = get_transformations(db_name, config.TASKS_CONFIG)

    if data_split_manifest is None or data_split_manifest.get("mode") == "official_val":
        dataset_train = get_mtl_dataset(db_name, config, train_transforms, split="train")
        dataset_val = get_mtl_dataset(db_name, config, val_transforms, split="val")
    else:
        train_dataset_full = get_mtl_dataset(db_name, config, train_transforms, split="train")
        val_dataset_full = get_mtl_dataset(db_name, config, val_transforms, split="train")
        dataset_train = Subset(train_dataset_full, data_split_manifest["meta_train_indices"])
        dataset_val = Subset(val_dataset_full, data_split_manifest["meta_val_indices"])

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=config.DATA.NUM_WORKERS,
        collate_fn=collate_mil,
        pin_memory=config.DATA.PIN_MEMORY,
    )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=config.DATA.NUM_WORKERS,
        collate_fn=collate_mil,
        pin_memory=config.DATA.PIN_MEMORY,
    )
    return dataset_train, dataset_val, data_loader_train, data_loader_val, None


def clone_state_dict_to_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def build_loss_bundle(config: CN):
    loss_ft = nn.ModuleDict(
        {task: get_loss(config["TASKS_CONFIG"], task, config) for task in config.TASKS}
    )
    loss_weights = {task: DEFAULT_TASK_LOSS_WEIGHTS[task] for task in config.TASKS}
    criterion = MultiTaskLoss(config.TASKS, loss_ft, loss_weights)
    return criterion, loss_ft, loss_weights


def move_batch_to_device(batch: Dict, tasks: Sequence[str], device: torch.device):
    samples = batch["image"].to(device, non_blocking=True)
    targets = {task: batch[task].to(device, non_blocking=True) for task in tasks}
    return samples, targets


def maybe_load_initial_weights(config: CN, model: nn.Module, logger) -> None:
    old_eval_mode = bool(config.EVAL_MODE)
    config.defrost()
    config.EVAL_MODE = True
    config.freeze()
    try:
        if config.MODEL.RESUME:
            load_checkpoint(config, model, None, None, None, logger, quiet=True)
        elif config.MODEL.RESUME_BACKBONE:
            target_model = model.backbone if hasattr(model, "backbone") else model
            load_checkpoint(config, target_model, None, None, None, logger, backbone=True, quiet=True)
        elif config.MODEL.PRETRAINED:
            load_pretrained(config, model, logger)
    finally:
        config.defrost()
        config.EVAL_MODE = old_eval_mode
        config.freeze()


def build_task_model(config: CN, device: torch.device, logger, init_state_dict: Dict[str, torch.Tensor] = None):
    model = build_model(config)
    if config.MTL:
        model = build_mtl_model(model, config)
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict, strict=False)
    else:
        maybe_load_initial_weights(config, model, logger)
    model.to(device)
    if config.MODEL.TADMTL.ENABLED and config.MODEL.TADMTL.FREEZE_PRETRAINED:
        mark_only_lora_as_trainable(
            model.backbone,
            bias=config.MODEL.TADMTL.BIAS,
            freeze_patch_embed=config.TRAIN.FREEZE_PATCH_EMBED,
            freeze_norm=config.TRAIN.FREEZE_LAYER_NORM,
            free_relative_bias=config.TRAIN.FREEZE_RELATIVE_POSITION_BIAS,
            freeze_downsample_reduction=(
                True if config.MODEL.TADMTL.DOWNSAMPLER_ENABLED else config.TRAIN.FREEZE_DOWNSAMPLE_REDUCTION
            ),
        )
    return model


def evaluate_task_losses(model, data_loader, loss_ft, tasks, device: torch.device, max_batches: int = None):
    model.eval()
    aggregated = {task: 0.0 for task in tasks}
    num_batches = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            samples, targets = move_batch_to_device(batch, tasks, device)
            outputs = model(samples)
            for task in tasks:
                aggregated[task] += float(loss_ft[task](outputs[task], targets[task]).item())
            num_batches += 1
    if num_batches == 0:
        return {task: 0.0 for task in tasks}
    return {task: value / float(num_batches) for task, value in aggregated.items()}


def get_shared_ta_parameters(model: nn.Module):
    params = []
    for name, param in model.named_parameters():
        if "backbone" not in name:
            continue
        if "lora_shared_" not in name:
            continue
        if "groups" in name:
            continue
        params.append(param)
    if not params:
        raise ValueError("No baseline shared TA-LoRA parameters were found for Stage-1 affinity collection.")
    return params


def flatten_gradient_list(params, grads):
    flattened = []
    for param, grad in zip(params, grads):
        if grad is None:
            flattened.append(torch.zeros_like(param).reshape(-1))
        else:
            flattened.append(grad.detach().reshape(-1))
    return torch.cat(flattened, dim=0)


def build_pseudo_update(flat_grad: torch.Tensor, optimizer) -> torch.Tensor:
    lr = float(optimizer.param_groups[0]["lr"])
    momentum = float(optimizer.param_groups[0].get("momentum", 0.0))
    if momentum <= 0.0:
        return lr * flat_grad

    buffers = []
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is None:
                continue
            state = optimizer.state.get(param, {})
            if "momentum_buffer" in state:
                buffers.append(state["momentum_buffer"].detach().reshape(-1))
            else:
                buffers.append(torch.zeros_like(param).reshape(-1))
    if not buffers:
        return lr * flat_grad
    momentum_buffer = torch.cat(buffers, dim=0).to(flat_grad.device)
    if momentum_buffer.shape[0] != flat_grad.shape[0]:
        return lr * flat_grad
    return lr * (flat_grad + momentum * momentum_buffer)


def safe_num_batches(data_loader) -> int:
    try:
        return int(len(data_loader))
    except TypeError:
        return 0


def resolve_log_interval(num_batches: int) -> int:
    if num_batches <= 0:
        return 1
    return max(1, min(50, num_batches // 10 if num_batches >= 10 else 1))


def maybe_log_progress(logger, label: str, batch_idx: int, num_batches: int, log_interval: int, extra: str = "") -> None:
    if batch_idx == 1 or batch_idx == num_batches or batch_idx % max(1, log_interval) == 0:
        suffix = f" | {extra}" if extra else ""
        logger.info("%s | batch=%d/%d%s", label, batch_idx, num_batches, suffix)


def warmup_and_collect_affinity(config: CN, logger, working_dir: str, data_split_manifest: Dict = None):
    partition_granularity = normalize_partition_granularity(config.MODEL.AGMTLORA.PARTITION_GRANULARITY)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, data_loader_train, data_loader_val, _ = build_stage1_data_loaders(
        config,
        data_split_manifest=data_split_manifest,
    )
    model = build_task_model(config, device, logger)
    criterion, loss_ft, _ = build_loss_bundle(config)
    optimizer = build_optimizer(config, model)
    num_train_batches = safe_num_batches(data_loader_train)
    log_interval = resolve_log_interval(num_train_batches)
    warmup_epochs = int(config.MODEL.AGMTLORA.AFFINITY_WARMUP_EPOCHS)
    affinity_score_epochs = int(config.MODEL.AGMTLORA.AFFINITY_SCORE_EPOCHS)

    logger.info(
        "Stage-1 affinity start | tasks=%s | warmup_epochs=%d | affinity_score_epochs=%d | train_batches=%d",
        list(config.TASKS),
        warmup_epochs,
        affinity_score_epochs,
        num_train_batches,
    )

    for epoch_idx in range(warmup_epochs):
        model.train()
        for batch_idx, batch in enumerate(data_loader_train, start=1):
            samples, targets = move_batch_to_device(batch, config.TASKS, device)
            optimizer.zero_grad()
            outputs = model(samples)
            loss, _ = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            maybe_log_progress(
                logger,
                f"Warmup epoch {epoch_idx + 1}/{warmup_epochs}",
                batch_idx,
                num_train_batches,
                log_interval,
                extra=f"loss={float(loss.item()):.6f}",
            )

    warmup_validation_losses = evaluate_task_losses(model, data_loader_val, loss_ft, config.TASKS, device)
    warmup_state_dict = clone_state_dict_to_cpu(model)
    warmup_checkpoint_path = os.path.join(working_dir, "warmup_checkpoint.pth")
    mkdir_if_missing(os.path.dirname(warmup_checkpoint_path))
    torch.save(
        {
            "model": warmup_state_dict,
            "epoch": max(0, warmup_epochs - 1),
            "extra_state": {
                "stage": "ag_tadformer_stage1_warmup",
                "tasks": list(config.TASKS),
                "affinity_warmup_epochs": warmup_epochs,
                "partition_granularity": partition_granularity,
            },
        },
        warmup_checkpoint_path,
    )

    shared_params = get_shared_ta_parameters(model)
    task_index = {task: idx for idx, task in enumerate(config.TASKS)}
    affinity_epoch_history = []
    num_batches_per_epoch = []

    for affinity_epoch_idx in range(affinity_score_epochs):
        model.train()
        epoch_directed_sum = torch.zeros((len(config.TASKS), len(config.TASKS)), dtype=torch.float32, device=device)
        epoch_batches = 0
        for batch_idx, batch in enumerate(data_loader_train, start=1):
            samples, targets = move_batch_to_device(batch, config.TASKS, device)
            optimizer.zero_grad()
            outputs = model(samples)

            flat_gradients = {}
            pseudo_updates = {}
            for task in config.TASKS:
                task_loss = loss_ft[task](outputs[task], targets[task])
                grads = torch.autograd.grad(
                    task_loss,
                    shared_params,
                    retain_graph=True,
                    allow_unused=True,
                )
                flat_grad = flatten_gradient_list(shared_params, grads)
                flat_gradients[task] = flat_grad
                pseudo_updates[task] = build_pseudo_update(flat_grad, optimizer)

            lr = float(optimizer.param_groups[0]["lr"])
            for src_task in config.TASKS:
                src_idx = task_index[src_task]
                for dst_task in config.TASKS:
                    dst_idx = task_index[dst_task]
                    dot_value = torch.dot(flat_gradients[dst_task], pseudo_updates[src_task])
                    epoch_directed_sum[src_idx, dst_idx] += dot_value / max(lr, 1e-12)

            epoch_batches += 1
            loss, _ = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            maybe_log_progress(
                logger,
                f"Affinity epoch {affinity_epoch_idx + 1}/{affinity_score_epochs}",
                batch_idx,
                num_train_batches,
                log_interval,
                extra=f"loss={float(loss.item()):.6f}",
            )

        epoch_directed_affinity = (epoch_directed_sum / max(float(epoch_batches), 1.0)).detach().cpu().numpy()
        affinity_epoch_history.append(epoch_directed_affinity)
        num_batches_per_epoch.append(int(epoch_batches))
        logger.info(
            "Affinity epoch %d/%d finished | batches=%d | mean=%.6f | std=%.6f",
            affinity_epoch_idx + 1,
            affinity_score_epochs,
            epoch_batches,
            float(np.mean(epoch_directed_affinity)),
            float(np.std(epoch_directed_affinity)),
        )

    if affinity_epoch_history:
        directed_affinity = np.mean(np.stack(affinity_epoch_history, axis=0), axis=0)
    else:
        directed_affinity = np.zeros((len(config.TASKS), len(config.TASKS)), dtype=np.float32)
    symmetric_affinity = 0.5 * (directed_affinity + directed_affinity.T)
    post_affinity_validation_losses = evaluate_task_losses(model, data_loader_val, loss_ft, config.TASKS, device)
    post_affinity_state_dict = clone_state_dict_to_cpu(model)
    post_affinity_checkpoint_path = os.path.join(working_dir, "post_affinity_checkpoint.pth")
    torch.save(
        {
            "model": post_affinity_state_dict,
            "epoch": max(0, warmup_epochs + affinity_score_epochs - 1),
            "extra_state": {
                "stage": "ag_tadformer_stage1_post_affinity",
                "tasks": list(config.TASKS),
                "affinity_warmup_epochs": warmup_epochs,
                "affinity_score_epochs": affinity_score_epochs,
                "partition_granularity": partition_granularity,
            },
        },
        post_affinity_checkpoint_path,
    )
    logger.info("Stage-1 affinity finished | post_affinity_checkpoint=%s", post_affinity_checkpoint_path)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "warmup_checkpoint_path": warmup_checkpoint_path,
        "warmup_state_dict": warmup_state_dict,
        "warmup_validation_losses": warmup_validation_losses,
        "post_affinity_checkpoint_path": post_affinity_checkpoint_path,
        "post_affinity_state_dict": post_affinity_state_dict,
        "post_affinity_validation_losses": post_affinity_validation_losses,
        "directed_affinity": directed_affinity,
        "symmetric_affinity": symmetric_affinity,
        "affinity_epoch_history": affinity_epoch_history,
        "num_affinity_epochs": affinity_score_epochs,
        "num_batches_per_epoch": num_batches_per_epoch,
    }


def build_group_proxy(directed_affinity: np.ndarray, tasks: Sequence[str], candidate_groups: Sequence[Sequence[str]]):
    task_index = {task: idx for idx, task in enumerate(tasks)}
    group_proxy = {}
    csv_rows = []
    for group in candidate_groups:
        group = list(group)
        group_key = group_to_key(group)
        group_proxy[group_key] = {}
        for task in group:
            if len(group) == 1:
                proxy_value = 0.0
            else:
                incoming = [
                    float(directed_affinity[task_index[other_task], task_index[task]])
                    for other_task in group
                    if other_task != task
                ]
                proxy_value = float(np.mean(incoming)) if incoming else 0.0
            group_proxy[group_key][task] = proxy_value
            csv_rows.append({"group": group_key, "task": task, "proxy": proxy_value})
    return group_proxy, csv_rows


def run_partition_search(tasks: Sequence[str], search_scores: Dict[str, Dict[str, float]], max_groups: int):
    partitions = enumerate_partitions(tasks, max_groups)
    ranked_results = []
    for partition in partitions:
        partition = canonicalize_groups(partition, tasks)
        per_task_scores = {}
        for group in partition:
            group_key = group_to_key(group)
            for task in group:
                per_task_scores[task] = float(search_scores[group_key][task])
        score = float(np.mean([per_task_scores[task] for task in tasks]))
        ranked_results.append(
            {
                "groups": partition,
                "partition_score": score,
                "per_task_scores": per_task_scores,
                "num_groups": len(partition),
            }
        )

    ranked_results.sort(
        key=lambda item: (
            -item["partition_score"],
            item["num_groups"],
            [group_to_key(group) for group in item["groups"]],
        )
    )
    return ranked_results


def create_resolved_training_config(
    base_cfg_path: str,
    grouping_json_path: str,
    resolved_group_ranks,
    post_affinity_checkpoint_path: str,
    partition_granularity: str = "global",
) -> CN:
    if not base_cfg_path:
        raise ValueError("AG-TADFormer Stage-1 requires the original --cfg path to build a reusable training config.")
    resolved_config = CN(new_allowed=True)
    resolved_config.BASE = [os.path.abspath(base_cfg_path)]
    resolved_config.MODEL = CN(new_allowed=True)
    resolved_config.MODEL.RESUME = os.path.abspath(post_affinity_checkpoint_path)
    resolved_config.MODEL.AGMTLORA = CN(new_allowed=True)
    resolved_config.MODEL.AGMTLORA.ENABLED = True
    resolved_config.MODEL.AGMTLORA.STAGE = 1
    resolved_config.MODEL.AGMTLORA.PARTITION_GRANULARITY = normalize_partition_granularity(partition_granularity)
    resolved_config.MODEL.AGMTLORA.SEARCH_SCORE_SOURCE = SEARCH_SCORE_SOURCE_GROUP_PROXY
    resolved_config.MODEL.AGMTLORA.GROUPING_SOURCE = "fixed_json"
    resolved_config.MODEL.AGMTLORA.GROUPING_JSON = os.path.abspath(grouping_json_path)
    resolved_config.MODEL.AGMTLORA.GROUP_SHARED_RANKS = [
        [int(rank) for rank in stage_ranks] for stage_ranks in resolved_group_ranks
    ]
    resolved_config.TRAIN = CN(new_allowed=True)
    resolved_config.TRAIN.AUTO_RESUME = False
    resolved_config.freeze()
    return resolved_config


def _partition_results_csv_rows(ranked_partitions):
    rows = []
    for rank_idx, result in enumerate(ranked_partitions):
        rows.append(
            {
                "rank": rank_idx + 1,
                "partition_score": result["partition_score"],
                "num_groups": result["num_groups"],
                "groups": json.dumps(result["groups"], ensure_ascii=False),
                "per_task_scores": json.dumps(result["per_task_scores"], ensure_ascii=False),
            }
        )
    return rows


def write_search_artifacts(
    *,
    tasks: Sequence[str],
    search_scores,
    search_score_source: str,
    output_root: str,
    grouping_save_path: str,
    max_groups: int,
    group_shared_ranks,
    total_shared_rank_budget: int,
    num_stages: int,
    group_rank_allocation: str,
    search_objective: str,
    affinity_path: str,
    warmup_checkpoint_path: str,
    post_affinity_checkpoint_path: str,
    affinity_warmup_epochs: int,
    affinity_score_epochs: int,
    base_cfg_path: str,
    runtime_snapshot_text: str,
    logger,
    partition_granularity: str = "global",
) -> Dict[str, object]:
    search_score_source = normalize_search_score_source(search_score_source)
    partition_granularity = normalize_partition_granularity(partition_granularity)

    ranked_partitions = run_partition_search(tasks, search_scores, int(max_groups))
    if not ranked_partitions:
        raise ValueError("No valid task partitions were generated.")
    best_partition = ranked_partitions[0]
    best_groups = best_partition["groups"]
    group_names = [group_display_name(group_idx) for group_idx in range(len(best_groups))]
    task_to_group = build_task_to_group(best_groups, group_names)
    resolved_group_ranks, rank_source = resolve_group_shared_ranks(
        group_shared_ranks,
        int(total_shared_rank_budget),
        len(best_groups),
        int(num_stages),
        str(group_rank_allocation),
    )

    partition_results_json_path = _artifact_path(output_root, "partition_search_results", search_score_source)
    partition_results_csv_path = _artifact_path(output_root, "partition_search_results", search_score_source, ".csv")
    grouping_json_path = append_search_score_suffix(grouping_save_path, search_score_source)
    resolved_config_path = _artifact_path(output_root, "resolved_agmtlora_config", search_score_source, ".yaml")

    save_json(
        {
            "tasks": list(tasks),
            "partition_granularity": partition_granularity,
            "search_score_source": search_score_source,
            "search_objective": search_objective,
            "ranked_partitions": ranked_partitions,
        },
        partition_results_json_path,
    )
    save_rows_csv(
        _partition_results_csv_rows(ranked_partitions),
        partition_results_csv_path,
        fieldnames=["rank", "partition_score", "num_groups", "groups", "per_task_scores"],
    )

    grouping_payload = {
        "tasks": list(tasks),
        "partition_granularity": partition_granularity,
        "search_score_source": search_score_source,
        "search_objective": search_objective,
        "groups": best_groups,
        "group_names": group_names,
        "task_to_group": task_to_group,
        "num_groups": len(best_groups),
        "group_shared_ranks": resolved_group_ranks,
        "group_rank_source": rank_source,
        "total_shared_rank_budget": int(total_shared_rank_budget),
        "best_partition": best_partition,
        "partition_search_results_path": partition_results_json_path,
        "search_score_path": os.path.join(output_root, "group_proxy.json"),
        "affinity_path": affinity_path,
        "warmup_checkpoint": warmup_checkpoint_path,
        "post_affinity_checkpoint": post_affinity_checkpoint_path,
        "affinity_warmup_epochs": int(affinity_warmup_epochs),
        "affinity_score_epochs": int(affinity_score_epochs),
    }
    save_json(grouping_payload, grouping_json_path)

    resolved_config = create_resolved_training_config(
        base_cfg_path=base_cfg_path,
        grouping_json_path=grouping_json_path,
        resolved_group_ranks=resolved_group_ranks,
        post_affinity_checkpoint_path=post_affinity_checkpoint_path,
        partition_granularity=partition_granularity,
    )
    mkdir_if_missing(os.path.dirname(resolved_config_path))
    with open(resolved_config_path, "w", encoding="utf-8") as handle:
        handle.write(resolved_config.dump())

    runtime_snapshot_path = _artifact_path(output_root, "runtime_config_snapshot", search_score_source, ".yaml")
    with open(runtime_snapshot_path, "w", encoding="utf-8") as handle:
        handle.write(runtime_snapshot_text)

    logger.info(
        "Stage-1 grouping selected | score=%.6f | groups=%s | grouping=%s | resolved_config=%s",
        float(best_partition["partition_score"]),
        best_groups,
        grouping_json_path,
        resolved_config_path,
    )
    return {
        "grouping_json": grouping_json_path,
        "resolved_config": resolved_config_path,
        "partition_search_results_json": partition_results_json_path,
        "partition_search_results_csv": partition_results_csv_path,
        "best_partition": best_partition,
        "group_shared_ranks": resolved_group_ranks,
        "group_rank_source": rank_source,
    }


def run_stage1_pipeline(config: CN, output_root: str, logger, base_cfg_path: str):
    mkdir_if_missing(output_root)
    normalize_partition_granularity(config.MODEL.AGMTLORA.PARTITION_GRANULARITY)
    search_score_source = get_search_score_source(config)
    logger.info("AG-TADFormer Stage-1 pipeline started | output_root=%s", output_root)

    data_split_manifest = build_stage1_data_split_manifest(config, logger)
    affinity_result = warmup_and_collect_affinity(
        config,
        logger,
        output_root,
        data_split_manifest=data_split_manifest,
    )

    affinity_json_path = os.path.abspath(config.MODEL.AGMTLORA.AFFINITY_SAVE_PATH)
    affinity_csv_path = os.path.splitext(affinity_json_path)[0] + ".csv"
    save_json(
        {
            "tasks": list(config.TASKS),
            "partition_granularity": config.MODEL.AGMTLORA.PARTITION_GRANULARITY,
            "search_score_source": search_score_source,
            "data_split_manifest": data_split_manifest,
            "directed_affinity": affinity_result["directed_affinity"].tolist(),
            "symmetric_affinity": affinity_result["symmetric_affinity"].tolist(),
            "affinity_epoch_history": [matrix.tolist() for matrix in affinity_result["affinity_epoch_history"]],
            "num_affinity_epochs": int(affinity_result["num_affinity_epochs"]),
            "num_batches_per_epoch": list(affinity_result["num_batches_per_epoch"]),
            "warmup_checkpoint": affinity_result["warmup_checkpoint_path"],
            "post_affinity_checkpoint": affinity_result["post_affinity_checkpoint_path"],
            "warmup_validation_losses": affinity_result["warmup_validation_losses"],
            "post_affinity_validation_losses": affinity_result["post_affinity_validation_losses"],
        },
        affinity_json_path,
    )
    save_matrix_csv(config.TASKS, affinity_result["directed_affinity"], affinity_csv_path)

    candidate_groups = enumerate_candidate_groups(config.TASKS)
    group_proxy, group_proxy_csv_rows = build_group_proxy(
        affinity_result["directed_affinity"],
        config.TASKS,
        candidate_groups,
    )
    group_proxy_json_path = os.path.join(output_root, "group_proxy.json")
    group_proxy_csv_path = os.path.join(output_root, "group_proxy.csv")
    save_json(
        {
            "tasks": list(config.TASKS),
            "partition_granularity": config.MODEL.AGMTLORA.PARTITION_GRANULARITY,
            "search_score_source": search_score_source,
            "candidate_groups": candidate_groups,
            "group_proxy": group_proxy,
            "affinity_path": affinity_json_path,
        },
        group_proxy_json_path,
    )
    save_rows_csv(group_proxy_csv_rows, group_proxy_csv_path, fieldnames=["group", "task", "proxy"])

    search_artifacts = write_search_artifacts(
        tasks=config.TASKS,
        search_scores=group_proxy,
        search_score_source=search_score_source,
        output_root=output_root,
        grouping_save_path=config.MODEL.AGMTLORA.GROUPING_SAVE_PATH,
        max_groups=int(config.MODEL.AGMTLORA.MAX_GROUPS),
        group_shared_ranks=config.MODEL.AGMTLORA.GROUP_SHARED_RANKS,
        total_shared_rank_budget=int(config.MODEL.AGMTLORA.TOTAL_SHARED_RANK_BUDGET),
        num_stages=len(config.MODEL.SWIN.DEPTHS),
        group_rank_allocation=str(config.MODEL.AGMTLORA.GROUP_RANK_ALLOCATION),
        search_objective=str(config.MODEL.AGMTLORA.SEARCH_OBJECTIVE),
        affinity_path=affinity_json_path,
        warmup_checkpoint_path=affinity_result["warmup_checkpoint_path"],
        post_affinity_checkpoint_path=affinity_result["post_affinity_checkpoint_path"],
        affinity_warmup_epochs=int(config.MODEL.AGMTLORA.AFFINITY_WARMUP_EPOCHS),
        affinity_score_epochs=int(config.MODEL.AGMTLORA.AFFINITY_SCORE_EPOCHS),
        base_cfg_path=base_cfg_path,
        runtime_snapshot_text=config.dump(),
        logger=logger,
        partition_granularity=config.MODEL.AGMTLORA.PARTITION_GRANULARITY,
    )

    artifacts = {
        "output_root": output_root,
        "affinity_json": affinity_json_path,
        "affinity_csv": affinity_csv_path,
        "group_proxy_json": group_proxy_json_path,
        "group_proxy_csv": group_proxy_csv_path,
        "warmup_checkpoint": affinity_result["warmup_checkpoint_path"],
        "post_affinity_checkpoint": affinity_result["post_affinity_checkpoint_path"],
    }
    artifacts.update(search_artifacts)
    return artifacts

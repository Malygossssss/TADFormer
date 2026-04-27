import itertools
import json
import os
from typing import Dict, Iterable, List, Sequence, Tuple


DEFAULT_PARTITION_GRANULARITY = "global"
SUPPORTED_PARTITION_GRANULARITIES = {"global"}


def group_display_name(group_index: int) -> str:
    return f"group_{int(group_index)}"


def normalize_partition_granularity(partition_granularity: str) -> str:
    granularity = str(partition_granularity or DEFAULT_PARTITION_GRANULARITY)
    if granularity not in SUPPORTED_PARTITION_GRANULARITIES:
        raise ValueError(
            f"Unsupported PARTITION_GRANULARITY: {granularity}. "
            "This TADFormer port currently supports only 'global'."
        )
    return granularity


def _task_order_map(tasks: Sequence[str]) -> Dict[str, int]:
    return {task: idx for idx, task in enumerate(tasks)}


def canonicalize_groups(groups: Iterable[Iterable[str]], tasks: Sequence[str]) -> List[List[str]]:
    task_order = _task_order_map(tasks)
    normalized = []
    for group in groups:
        unique_group = []
        seen = set()
        for task in group:
            if task not in task_order:
                raise ValueError(f"Unknown task in grouping: {task}")
            if task in seen:
                continue
            seen.add(task)
            unique_group.append(task)
        normalized.append(sorted(unique_group, key=lambda task: task_order[task]))
    normalized.sort(key=lambda group: [task_order[task] for task in group])
    return normalized


def build_task_to_group(groups: Sequence[Sequence[str]], group_names: Sequence[str] = None) -> Dict[str, str]:
    task_to_group = {}
    if group_names is None:
        group_names = [group_display_name(group_idx) for group_idx in range(len(groups))]
    if len(group_names) < len(groups):
        raise ValueError("Not enough group names were provided for the requested groups.")
    for group_idx, group in enumerate(groups):
        group_name = str(group_names[group_idx])
        for task in group:
            if task in task_to_group:
                raise ValueError(f"Task '{task}' appears in multiple groups.")
            task_to_group[task] = group_name
    return task_to_group


def _validate_groups_cover_tasks(groups: Sequence[Sequence[str]], expected_tasks: Sequence[str]) -> None:
    task_order = _task_order_map(expected_tasks)
    covered_tasks = sorted(
        itertools.chain.from_iterable(groups),
        key=lambda task: task_order[task],
    )
    if list(expected_tasks) != covered_tasks:
        raise ValueError(
            "Grouping JSON tasks do not match config tasks. "
            f"Expected {list(expected_tasks)}, got {covered_tasks}."
        )


def load_grouping_json(grouping_json_path: str, expected_tasks: Sequence[str]) -> Dict:
    with open(grouping_json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    partition_granularity = normalize_partition_granularity(
        payload.get("partition_granularity", DEFAULT_PARTITION_GRANULARITY)
    )
    payload["partition_granularity"] = partition_granularity
    groups = canonicalize_groups(payload.get("groups", []), expected_tasks)
    _validate_groups_cover_tasks(groups, expected_tasks)
    group_names = [group_display_name(group_idx) for group_idx in range(len(groups))]
    payload["groups"] = groups
    payload["group_names"] = group_names
    payload["task_to_group"] = build_task_to_group(groups, group_names)
    payload["num_groups"] = len(groups)
    return payload


def resolve_group_shared_ranks(
    group_shared_ranks,
    total_shared_rank_budget: int,
    num_groups: int,
    num_stages: int,
    allocation: str = "equal_split",
) -> Tuple[List[List[int]], str]:
    if num_groups <= 0:
        return [], "manual"

    if group_shared_ranks:
        if all(isinstance(rank, int) for rank in group_shared_ranks):
            if len(group_shared_ranks) != num_groups:
                raise ValueError("GROUP_SHARED_RANKS must provide one rank per group.")
            return [[int(rank)] * int(num_stages) for rank in group_shared_ranks], "manual"
        if all(isinstance(rank, (list, tuple)) for rank in group_shared_ranks):
            if len(group_shared_ranks) != num_groups:
                raise ValueError("GROUP_SHARED_RANKS must provide one rank entry per group.")
            resolved = []
            for rank in group_shared_ranks:
                if len(rank) == 1:
                    resolved.append([int(rank[0])] * int(num_stages))
                elif len(rank) == int(num_stages):
                    resolved.append([int(v) for v in rank])
                else:
                    raise ValueError(
                        "Each group rank override must have length 1 or match the number of stages."
                    )
            return resolved, "manual"
        raise ValueError("GROUP_SHARED_RANKS must be a flat list or a nested list.")

    if str(allocation) != "equal_split":
        raise ValueError(f"Unsupported GROUP_RANK_ALLOCATION: {allocation}")

    base_rank = int(total_shared_rank_budget) // int(num_groups)
    remainder = int(total_shared_rank_budget) % int(num_groups)
    per_group = [base_rank + (1 if idx < remainder else 0) for idx in range(num_groups)]
    return [[rank] * int(num_stages) for rank in per_group], "auto_equal_split"


def enumerate_candidate_groups(tasks: Sequence[str]) -> List[List[str]]:
    candidates = []
    for group_size in range(1, len(tasks) + 1):
        for group in itertools.combinations(tasks, group_size):
            candidates.append(list(group))
    return candidates


def enumerate_partitions(tasks: Sequence[str], max_groups: int) -> List[List[List[str]]]:
    partitions: List[List[List[str]]] = []

    def _helper(task_index: int, current_partition: List[List[str]]) -> None:
        if task_index == len(tasks):
            partitions.append([group[:] for group in current_partition])
            return

        task = tasks[task_index]
        for group in current_partition:
            group.append(task)
            _helper(task_index + 1, current_partition)
            group.pop()

        if len(current_partition) < int(max_groups):
            current_partition.append([task])
            _helper(task_index + 1, current_partition)
            current_partition.pop()

    _helper(0, [])
    return [canonicalize_groups(partition, tasks) for partition in partitions]


def resolve_artifact_path(base_output_dir: str, path_value: str, default_name: str) -> str:
    path_value = str(path_value or "").strip()
    if not path_value:
        path_value = os.path.join(base_output_dir, default_name)
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(base_output_dir, path_value))


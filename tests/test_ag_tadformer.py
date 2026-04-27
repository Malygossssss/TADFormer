import logging
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from yacs.config import CfgNode as CN

from ag_mtlora.config_utils import (
    canonicalize_groups,
    enumerate_candidate_groups,
    enumerate_partitions,
    resolve_group_shared_ranks,
)
from ag_mtlora.stage1 import build_group_proxy, run_partition_search, run_stage1_pipeline
from config import get_config
from models.lora import TAModuleLinear
from utils import _expand_agmtlora_shared_state


def test_group_helpers_and_rank_split():
    tasks = ["semseg", "normals", "sal"]
    assert canonicalize_groups([["sal", "semseg", "sal"], ["normals"]], tasks) == [
        ["semseg", "sal"],
        ["normals"],
    ]
    assert enumerate_candidate_groups(tasks) == [
        ["semseg"],
        ["normals"],
        ["sal"],
        ["semseg", "normals"],
        ["semseg", "sal"],
        ["normals", "sal"],
        ["semseg", "normals", "sal"],
    ]
    partitions = enumerate_partitions(tasks, max_groups=2)
    assert [["semseg", "normals", "sal"]] in partitions
    assert [["semseg", "normals"], ["sal"]] in partitions

    ranks, source = resolve_group_shared_ranks([], total_shared_rank_budget=7, num_groups=3, num_stages=2)
    assert source == "auto_equal_split"
    assert ranks == [[3, 3], [2, 2], [2, 2]]


def test_group_proxy_and_partition_ranking():
    tasks = ["a", "b", "c"]
    directed_affinity = np.array(
        [
            [0.0, 3.0, 1.0],
            [4.0, 0.0, 2.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    candidate_groups = enumerate_candidate_groups(tasks)
    group_proxy, rows = build_group_proxy(directed_affinity, tasks, candidate_groups)
    assert group_proxy["a"]["a"] == 0.0
    assert group_proxy["a+b"]["a"] == 4.0
    assert group_proxy["a+b"]["b"] == 3.0
    assert len(rows) == sum(len(group) for group in candidate_groups)

    ranked = run_partition_search(tasks, group_proxy, max_groups=2)
    assert ranked[0]["groups"] == [["a", "b"], ["c"]]
    assert ranked[0]["partition_score"] == pytest.approx((4.0 + 3.0 + 0.0) / 3.0)


def _args_for_config(cfg_path, output, grouping_json=None, search_score_source=None):
    opts = []
    if grouping_json:
        opts.extend(["MODEL.AGMTLORA.GROUPING_JSON", grouping_json])
    if search_score_source:
        opts.extend(["MODEL.AGMTLORA.SEARCH_SCORE_SOURCE", search_score_source])
    return SimpleNamespace(
        cfg=cfg_path,
        opts=opts or None,
        batch_size=None,
        ckpt_freq=None,
        eval_freq=None,
        epochs=None,
        data_path=None,
        zip=False,
        cache_mode=None,
        pretrained=None,
        resume=None,
        accumulation_steps=None,
        use_checkpoint=False,
        disable_amp=False,
        output=str(output),
        name=None,
        tag=None,
        eval=False,
        throughput=False,
        local_rank=0,
        fused_window_process=False,
        fused_layernorm=False,
        optim=None,
        tasks="semseg,normals",
        nyud=None,
        pascal=str(output / "pascal"),
        eval_training_freq=None,
        resume_backbone=None,
        freeze_backbone=False,
        skip_initial_validation=False,
        decoder_map=None,
        skip_decoder=False,
    )


def test_fixed_json_config_resolution_and_invalid_source(tmp_path):
    grouping_path = tmp_path / "grouping.json"
    grouping_path.write_text(
        '{"partition_granularity": "global", "groups": [["normals"], ["semseg"]]}',
        encoding="utf-8",
    )
    cfg_path = tmp_path / "ag_fixed.yaml"
    base_cfg = os.path.abspath("configs/TADFormer/TADFormer_r64_Swin-T.yaml")
    cfg_path.write_text(
        f"""
BASE: ['{base_cfg}']
MODEL:
  AGMTLORA:
    ENABLED: True
    STAGE: 1
    GROUPING_SOURCE: fixed_json
    TOTAL_SHARED_RANK_BUDGET: 8
""",
        encoding="utf-8",
    )

    config = get_config(_args_for_config(str(cfg_path), tmp_path, str(grouping_path)))
    assert config.MODEL.TADMTL.AGMTLORA_ENABLED
    assert config.MODEL.TADMTL.AGMTLORA_GROUPS == [["semseg"], ["normals"]]
    assert config.MODEL.TADMTL.AGMTLORA_GROUP_RANKS == [[4, 4, 4, 4], [4, 4, 4, 4]]
    assert config.MODEL.TADMTL.AGMTLORA_TASK_TO_GROUP.semseg == "group_0"
    assert config.MODEL.TADMTL.AGMTLORA_TASK_TO_GROUP.normals == "group_1"

    with pytest.raises(ValueError, match="group_proxy"):
        get_config(_args_for_config(str(cfg_path), tmp_path, str(grouping_path), "final_predictions"))


def test_tamodulelinear_ag_route_smoke():
    module = TAModuleLinear(
        4,
        3,
        r={"group_0": 2, "group_1": 1},
        tasks=["a", "b"],
        task_to_group={"a": "group_0", "b": "group_1"},
        enabled=True,
    )
    x = torch.randn(2, 5, 4)
    pretrained, task_outputs = module(x)
    assert pretrained.shape == (2, 5, 3)
    assert set(module.lora_shared_A_groups.keys()) == {"group_0", "group_1"}
    assert task_outputs["a"].shape == (2, 5, 3)
    assert task_outputs["b"].shape == (2, 5, 3)

    baseline = TAModuleLinear(4, 3, r=2, tasks=["a", "b"], enabled=True)
    baseline_pretrained, baseline_tasks = baseline(x)
    assert baseline_pretrained.shape == (2, 5, 3)
    assert baseline_tasks["a"].shape == (2, 5, 3)


def test_checkpoint_ag_expansion_rank_intersection():
    cfg = CN(new_allowed=True)
    cfg.MODEL = CN(new_allowed=True)
    cfg.MODEL.TADMTL = CN(new_allowed=True)
    cfg.MODEL.TADMTL.AGMTLORA_ENABLED = True
    cfg.MODEL.TADMTL.AGMTLORA_GROUP_NAMES = ["group_0", "group_1"]
    source = {
        "m.lora_shared_A": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "m.lora_shared_B": torch.arange(15, dtype=torch.float32).reshape(5, 3),
        "m.lora_shared_scale": torch.tensor([2.0]),
    }
    target = {
        "m.lora_shared_A_groups.group_0": torch.zeros(2, 4),
        "m.lora_shared_A_groups.group_1": torch.zeros(1, 4),
        "m.lora_shared_B_groups.group_0": torch.zeros(5, 2),
        "m.lora_shared_B_groups.group_1": torch.zeros(5, 1),
        "m.lora_shared_scale_groups.group_0": torch.zeros(1),
        "m.lora_shared_scale_groups.group_1": torch.zeros(1),
    }
    expanded = _expand_agmtlora_shared_state(source, target, cfg)
    assert torch.equal(expanded["m.lora_shared_A_groups.group_0"], source["m.lora_shared_A"][:2])
    assert torch.equal(expanded["m.lora_shared_A_groups.group_1"], source["m.lora_shared_A"][:1])
    assert torch.equal(expanded["m.lora_shared_B_groups.group_0"], source["m.lora_shared_B"][:, :2])
    assert torch.equal(expanded["m.lora_shared_B_groups.group_1"], source["m.lora_shared_B"][:, :1])
    assert torch.equal(expanded["m.lora_shared_scale_groups.group_0"], torch.tensor([2.0]))


class _TinyLoss(torch.nn.Module):
    def forward(self, output, target):
        return torch.mean((output - target) ** 2)


class _TinyCriterion:
    def __init__(self, tasks):
        self.tasks = list(tasks)
        self.losses = torch.nn.ModuleDict({task: _TinyLoss() for task in self.tasks})

    def __call__(self, outputs, targets):
        loss_dict = {task: self.losses[task](outputs[task], targets[task]) for task in self.tasks}
        return sum(loss_dict.values()), loss_dict


class _TinyStage1Model(torch.nn.Module):
    def __init__(self, tasks):
        super().__init__()
        self.tasks = list(tasks)
        self.backbone = torch.nn.Module()
        self.backbone.lora_shared_A = torch.nn.Parameter(torch.tensor([[0.5]], dtype=torch.float32))

    def forward(self, samples):
        scalar = samples.mean() * self.backbone.lora_shared_A.sum()
        return {
            task: scalar.reshape(1, 1, 1, 1).expand(samples.shape[0], 1, 1, 1)
            for task in self.tasks
        }


def _stage1_cfg(tmp_path):
    cfg = CN(new_allowed=True)
    cfg.TASKS = ["a", "b"]
    cfg.SEED = 1
    cfg.AMP_ENABLE = False
    cfg.MODEL = CN(new_allowed=True)
    cfg.MODEL.SWIN = CN(new_allowed=True)
    cfg.MODEL.SWIN.DEPTHS = [1, 1]
    cfg.MODEL.AGMTLORA = CN(new_allowed=True)
    cfg.MODEL.AGMTLORA.PARTITION_GRANULARITY = "global"
    cfg.MODEL.AGMTLORA.SEARCH_SCORE_SOURCE = "group_proxy"
    cfg.MODEL.AGMTLORA.AFFINITY_SAVE_PATH = str(tmp_path / "affinity.json")
    cfg.MODEL.AGMTLORA.GROUPING_SAVE_PATH = str(tmp_path / "grouping.json")
    cfg.MODEL.AGMTLORA.AFFINITY_WARMUP_EPOCHS = 1
    cfg.MODEL.AGMTLORA.AFFINITY_SCORE_EPOCHS = 1
    cfg.MODEL.AGMTLORA.MAX_GROUPS = 2
    cfg.MODEL.AGMTLORA.GROUP_SHARED_RANKS = []
    cfg.MODEL.AGMTLORA.TOTAL_SHARED_RANK_BUDGET = 4
    cfg.MODEL.AGMTLORA.GROUP_RANK_ALLOCATION = "equal_split"
    cfg.MODEL.AGMTLORA.SEARCH_OBJECTIVE = "mean_group_proxy"
    return cfg


def test_stage1_dry_smoke_artifacts(tmp_path, monkeypatch):
    import ag_mtlora.stage1 as stage1

    cfg = _stage1_cfg(tmp_path)
    batch = {
        "image": torch.ones(2, 1, 1, 1),
        "a": torch.ones(2, 1, 1, 1),
        "b": torch.zeros(2, 1, 1, 1),
    }
    model = _TinyStage1Model(cfg.TASKS)
    criterion = _TinyCriterion(cfg.TASKS)

    monkeypatch.setattr(stage1, "build_stage1_data_split_manifest", lambda config, logger: {"mode": "fake"})
    monkeypatch.setattr(stage1, "build_stage1_data_loaders", lambda config, data_split_manifest=None: (None, None, [batch], [batch], None))
    monkeypatch.setattr(stage1, "build_task_model", lambda config, device, logger: model.to(device))
    monkeypatch.setattr(stage1, "build_optimizer", lambda config, model: torch.optim.SGD(model.parameters(), lr=0.1))
    monkeypatch.setattr(stage1, "build_loss_bundle", lambda config: (criterion, criterion.losses, {"a": 1.0, "b": 1.0}))

    logger = logging.getLogger("test_stage1_dry_smoke_artifacts")
    artifacts = run_stage1_pipeline(cfg, str(tmp_path), logger, base_cfg_path=str(tmp_path / "base.yaml"))

    assert os.path.exists(artifacts["affinity_json"])
    assert os.path.exists(artifacts["group_proxy_json"])
    assert os.path.exists(artifacts["grouping_json"])
    assert os.path.exists(artifacts["resolved_config"])
    assert os.path.exists(artifacts["warmup_checkpoint"])
    assert os.path.exists(artifacts["post_affinity_checkpoint"])

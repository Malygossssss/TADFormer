# TADFormer : Task-Adaptive Dynamic Transformer for Efficient Multi-Task Learning


This is the official implementation of the paper: [TADFormer : Task-Adaptive Dynamic Transformer for Efficient Multi-Task Learning](https://arxiv.org/pdf/2501.04293) based on the [MTLoRA](https://github.com/scale-lab/MTLoRA) 


## Run TADFormer

1. Clone the repository
   ```bash
   git clone git@github.com:Min100KM/TADFormer.git
   cd TADFormer
   ```
2. Install requirements
   - Install `PyTorch>=1.12.0` and `torchvision>=0.13.0` with `CUDA>=11.6`
   - Install dependencies: `pip install -r requirements.txt`

3. Running TADFormer code:

**Run the code**
```
CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch --nproc_per_node 1 --master_port 29561 main.py --cfg configs/TADFormer/TADFormer_r64_Swin-T_nyuv2.yaml --nyud NYUv2_MT   --tasks semseg,normals,depth --batch-size 12 --ckpt-freq=20 --eval-freq=5 --epochs=300 --resume-backbone backbone/Swin/swin_tiny_patch4_window7_224.pth
```

**Eval**
```
python -m torch.distributed.launch --nproc_per_node 1 --master_port=12345 \
main.py --cfg configs/TADFormer/[config_name].yaml \
--pascal [pascal_dataset] --tasks semseg,normals,sal,human_parts \   
--batch-size 32 --ckpt-freq=20 --epoch=300 --resume [.pth path] \
--eval \
--disable_wandb
```

## Run AG-TADFormer Global Group-Proxy

This repository includes the AG-TADFormer port of the AG-MTLoRA search path for:

- `MODEL.AGMTLORA.PARTITION_GRANULARITY: global`
- `MODEL.AGMTLORA.SEARCH_SCORE_SOURCE: group_proxy`

Other AG-MTLoRA branches, such as stage-wise partitioning, predictor-chain search, and `final_predictions` search, are intentionally not implemented.

### Stage 1: Search Task Groups

Stage 1 trains a baseline TADFormer for a short warmup, collects directed affinity on the shared TA-LoRA parameters, builds group-proxy scores, and writes the selected global grouping.

Example for PASCAL-Context:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/ag_mtlora_stage1_prepare.py \
  --cfg configs/TADFormer/AG_TADFormer_stage1_r64_Swin-T.yaml \
  --pascal /path/to/PASCAL_MT \
  --tasks semseg,normals,sal,human_parts \
  --batch-size 12 \
  --resume-backbone backbone/Swin/swin_tiny_patch4_window7_224.pth \
  --output output \
  --tag ag_stage1 \
  --disable_amp \
  --opts MODEL.AGMTLORA.AFFINITY_WARMUP_EPOCHS 5 MODEL.AGMTLORA.AFFINITY_SCORE_EPOCHS 5 MODEL.AGMTLORA.MAX_GROUPS 3
```

Important Stage-1 outputs are written under:

```text
output/AG_TADFormer_stage1_r64_Swin-T/ag_stage1/ag_mtlora_stage1_prepare/run_YYYYMMDD_HHMMSS/
```

Key artifacts:

```text
affinity.json
group_proxy.json
group_proxy.csv
partition_search_results__group_proxy.json
partition_search_results__group_proxy.csv
grouping__group_proxy.json
resolved_agmtlora_config__group_proxy.yaml
warmup_checkpoint.pth
post_affinity_checkpoint.pth
```

### Stage 2: Train With The Searched Groups

Use the generated `resolved_agmtlora_config__group_proxy.yaml` directly. It points to `grouping__group_proxy.json` and initializes from `post_affinity_checkpoint.pth`.

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 29561 main.py \
  --cfg output/AG_TADFormer_stage1_r64_Swin-T/ag_stage1/ag_mtlora_stage1_prepare/run_YYYYMMDD_HHMMSS/resolved_agmtlora_config__group_proxy.yaml \
  --pascal /path/to/PASCAL_MT \
  --tasks semseg,normals,sal,human_parts \
  --batch-size 12 \
  --ckpt-freq 20 \
  --eval-freq 5 \
  --epochs 300 \
  --disable_wandb
```

When loading the Stage-1 checkpoint into the Stage-2 AG model, the old single shared TA-LoRA weights are expanded into every group-specific TA-LoRA bank with rank-aligned copying.

## Citation

```
@InProceedings{Baek_2025_CVPR,
    author    = {Baek, Seungmin and Lee, Soyul and Jo, Hayeon and Choi, Hyesong and Min, Dongbo},
    title     = {TADFormer: Task-Adaptive Dynamic TransFormer for Efficient Multi-Task Learning},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {14858-14868}
}
```

## Acknowlegment

This repo benefits from the [MTLoRA](https://github.com/scale-lab/MTLoRA) and [ddfnet](https://github.com/theFoxofSky/ddfnet).


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
CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch --nproc_per_node 1 --master_port 29561 main.py --cfg configs/TADFormer/TADFormer_r64_Swin-T_nyuv2.yaml --nyud NYUv2_MT   --tasks semseg,normals,depth --batch-size 12 --ckpt-freq=20 --eval-freq=5 --epochs=300 --resume-backbone backbone/Swin/swin_tiny_patch4_window7_224.pth --disable-delta-mtl
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


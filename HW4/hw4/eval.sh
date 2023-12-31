#!bin/bash
python3 eval.py \
    --dataset_name self-defined \
    --root_dir ../dataset \
    --scene_name prediction \
    --split val \
    --N_importance 64 --img_wh 256 256 \
    --nerf_D 8 --nerf_W 256 \
    --save_depth --depth_format rgb \
    --ckpt_path checkpoints/8_256.ckpt \
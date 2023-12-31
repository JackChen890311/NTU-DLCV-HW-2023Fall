#!/bin/bash

# TODO - run your inference Python3 code
python3 hw4/eval.py \
    --dataset_name self-defined \
    --root_dir $1 \
    --scene_name $2 \
    --split test \
    --N_importance 64 --img_wh 256 256 \
    --nerf_D 8 --nerf_W 256 \
    --ckpt_path 8_256.ckpt \
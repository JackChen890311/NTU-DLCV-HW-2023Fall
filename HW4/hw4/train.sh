#!bin/bash
python3 train.py \
    --dataset_name self-defined \
    --root_dir ../dataset \
    --N_importance 64 --img_wh 256 256 --noise_std 0 \
    --num_epochs 10 --batch_size 1024 \
    --nerf_D 2 --nerf_W 64 \
    --optimizer adam --lr 5e-4 \
    --lr_scheduler steplr --decay_step 8 --decay_gamma 0.5 \
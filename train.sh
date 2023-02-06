#!/bin/bash

python train.py \
    --mode "test" \
    --dataset_name replica \
    --root_dir room_0/Sequence_1/ \
    --N_importance 64 \
    --img_wh 320 240 \
    --num_epochs 18 \
    --batch_size 1024 \
    --optimizer adam \
    --lr 5e-4 \
    --lr_scheduler steplr \
    --decay_step 4 8 \
    --decay_gamma 0.5 \
    --exp_name exp1-lc \
    --loss_type pm16 \
    --chunk 40000 \
    --nerf_model NeRFFeats \
    --num_gpus 1

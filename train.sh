python train.py --dataset_name llff_cls_ib --root_dir bowen_tou --N_importance 64 --img_wh 240 135 --num_epochs 30 --batch_size 1 --optimizer adam --lr 5e-4 --lr_scheduler steplr --decay_step 10 20 --decay_gamma 0.5 --exp_name debug --mode d3_ib --loss_type msenll --pretrained ./ckpts/exp-bowen-edit-point-0414-m1/epoch=19.ckpt
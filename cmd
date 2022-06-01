python train.py --dataset_name llff_cls_ib --root_dir bowen_tou --N_importance 64 --img_wh 480 270 --num_epochs 100 --batch_size 3 --optimizer adam --lr_scheduler steplr --decay_step 10 20 50 70 --decay_gamma 0.4 --mode d3_ib --loss_type msenll --pretrained epoch=19.ckpt --semantic_network conv3d --num_gpus 8 --lr 1e-3 --exp_name debug-0525-second-lr1e-3-e100-480x270-no-norm

python train.py --dataset_name blender_cls_ib --root_dir chair --N_importance 64 --img_wh 400 400 --num_epochs 100 --batch_size 3 --optimizer adam --lr_scheduler steplr --decay_step 10 20 50 70 --decay_gamma 0.4 --mode d3_ib --loss_type msenll --pretrained chair_pre.pth --semantic_network conv3d --num_gpus 2 --lr 1e-3 --exp_name debug-0531

python train.py --dataset_name llff_cls_ib --root_dir bowen_tou --N_importance 64 --img_wh 480 270 --num_epochs 100 --batch_size 3 --optimizer adam --lr_scheduler steplr --decay_step 10 20 50 70 --decay_gamma 0.4 --mode d3_ib --loss_type msenll --pretrained epoch=19.ckpt --semantic_network conv3d --num_gpus 2 --lr 1e-3 --exp_name debug-0532

python train.py --dataset_name llff_cls --root_dir bowen_tou --N_importance 64 --img_wh 640 360  --num_epochs 30 --batch_size 1024  --optimizer adam --lr 5e-4 --lr_scheduler steplr --decay_step 10 20 --decay_gamma 0.5 --exp_name debug  --loss_type mse --pretrained epoch\=19.ckpt

python train.py --dataset_name blender --root_dir chair --N_importance 64 --img_wh 400 400 --num_epochs 30 --batch_size 1024  --optimizer adam --lr 5e-4 --lr_scheduler steplr --decay_step 10 20 --decay_gamma 0.5 --exp_name debug  --loss_type mse --pretrained chair_pre.pth

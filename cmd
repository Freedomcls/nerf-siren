python train.py --dataset_name llff_cls_ib --root_dir bowen_tou --N_importance 64 --img_wh 480 270 --num_epochs 100 --batch_size 3 --optimizer adam --lr_scheduler steplr --decay_step 10 20 50 70 --decay_gamma 0.4 --mode d3_ib --loss_type msenll --pretrained epoch=19.ckpt --semantic_network conv3d --num_gpus 8 --lr 1e-3 --exp_name debug-0525-second-lr1e-3-e100-480x270-no-norm

python train.py --dataset_name blender_cls_ib --root_dir chair --N_importance 64 --img_wh 400 400 --num_epochs 100 --batch_size 3 --optimizer adam --lr_scheduler steplr --decay_step 10 20 50 70 --decay_gamma 0.4 --mode d3_ib --loss_type msenll --pretrained chair_pre.pth --semantic_network conv3d --num_gpus 2 --lr 1e-3 --exp_name debug-0531

python train.py --dataset_name blender_cls_ib --root_dir chair --N_importance 64 --img_wh 200 200 --num_epochs 100 --batch_size 3 --optimizer adam --lr_scheduler steplr --decay_step 10 20 50 70 --decay_gamma 0.4 --mode d3_ib --loss_type msenll --pretrained chair_pre.pth --semantic_network conv3d --num_gpus 2 --lr 1e-3 --exp_name debug-0531 --chunk 40000

python train.py --dataset_name llff_cls_ib --root_dir bowen_tou --N_importance 64 --img_wh 480 270 --num_epochs 100 --batch_size 3 --optimizer adam --lr_scheduler steplr --decay_step 10 20 50 70 --decay_gamma 0.4 --mode d3_ib --loss_type msenll --pretrained epoch=19.ckpt --semantic_network conv3d --num_gpus 2 --lr 1e-3 --exp_name debug-0532

python train.py --dataset_name llff_cls --root_dir bowen_tou --N_importance 64 --img_wh 640 360  --num_epochs 30 --batch_size 1024  --optimizer adam --lr 5e-4 --lr_scheduler steplr --decay_step 10 20 --decay_gamma 0.5 --exp_name debug  --loss_type mse --pretrained epoch\=19.ckpt

python train.py --dataset_name blender --root_dir chair --N_importance 64 --img_wh 400 400 --num_epochs 30 --batch_size 1024  --optimizer adam --lr 5e-4 --lr_scheduler steplr --decay_step 10 20 --decay_gamma 0.5 --exp_name debug  --loss_type mse --pretrained chair_pre.pth


椅子数据处理
训练
python train.py --dataset_name blender_cls_ib --root_dir chair --N_importance 64 --img_wh 400 400 --num_epochs 100 --batch_size 3 --optimizer adam --lr_scheduler steplr --decay_step 10 20 50 70 --decay_gamma 0.4 --mode d3_ib --loss_type msenll --pretrained chair_pre2.pth --semantic_network conv3d --num_gpus 8 --lr 1e-3 --exp_name debug-0607 --chunk 40000
测试 可视化验证集图像
CUDA_VISIBLE_DEVICES=7 python eval.py --mode d3_ib --root_dir chair --dataset_name blender_cls_ib --scene_name test_imgs --split test --img_wh 400 400 --N_importance 64 --chunk 40000 --ckpt_path ckpts/debug-0607/\{epoch\:d\}/epoch\=30-step\=154.ckpt --scene_name debug0607val -sn conv3d
测试 可视化带label的3D模型
1 首先生成训练集的label图像  python eval.py --mode d3_ib --root_dir chair --dataset_name blender_cls_ib --scene_name train_imgs --split train --img_wh 400 400 --N_importance 64 --chunk 40000 --ckpt_path ckpts/debug-0607/\{epoch\:d\}/epoch\=30-step\=154.ckpt --scene_name debug0607train -sn conv3d
2 在results/debug0607train下有生成label图像, 与训练集所用gt label图像名字对应, 把这些文件放到/data/chair/pre_labels文件夹下
3 然后利用label图像投影原理  python extract_color_mesh.py --root_dir chair/ --dataset_name blender --scene_name chair --img_wh 400 400 --ckpt_path ckpts/debug-0607/\{epoch\:d\}/epoch\=30-step\=154.ckpt  --sigma_threshold 20.0 --N_grid 256 --vis_type label

# color and seg end to end

python train.py --dataset_name blender_cls_ib --root_dir chair --N_importance 64 --img_wh 50 50 --num_epochs 130 --batch_size 1  --optimizer adam --lr 1e-4 --lr_scheduler steplr --decay_step 50 100 --decay_gamma 0.5 --exp_name debug_rgb  --loss_type mse --chunk 2500

python train.py --dataset_name blender_cls_ib --root_dir chair --N_importance 64 --img_wh 50 50 --num_epochs 130 --batch_size 1  --optimizer adam --lr 1e-4 --lr_scheduler steplr --decay_step 50 100 --decay_gamma 0.5 --exp_name debug_rgb_cls  --loss_type msenll --chunk 2500 --mode d3_ib --semantic_network conv3d --loss_type msenll --num_gpus 2 --pretrained ckpts/debug_rgb/\{epoch\:d\}/epoch\=129-step\=1689.ckpt


python eval.py  --root_dir chair --dataset_name blender --scene_name test_imgs --split test --img_wh 150 150 --N_importance 64 --chunk 22500 --ckpt_path ckpts/debug_memory2/\{epoch\:d\}/epoch\=299-step\=3899.ckpt

python train.py --dataset_name blender_cls_ib --root_dir chair --N_importance 64 --img_wh 150 150 --num_epochs 300 --batch_size 1  --optimizer adam --lr 1e-3 --lr_scheduler steplr --decay_step 100 200 --decay_gamma 0.5 --exp_name debug_memory2  --loss_type mse --chunk 2500 --num_gpus 8

python extract_color_mesh.py --root_dir chair/ --dataset_name blender --scene_name chair --img_wh 200 200 --ckpt_path ckpts/crop200x200_crop50x50_cls/\{epoch\:d\}/epoch\=299-step\=3899.ckpt  --sigma_threshold 20.0 --N_grid 256

# replica dataset

python train.py --dataset_name replica --root_dir room_0/Sequence_1/ --N_importance 64 --img_wh 320 240 --num_epochs 16 --batch_size 1024  --optimizer adam --lr 5e-4 --lr_scheduler steplr --decay_step 4 8 --decay_gamma 0.5 --exp_name debug_replica  --loss_type mse --chunk 40000

python eval.py  --root_dir room_0/Sequence_1/ --dataset_name replica --scene_name test_replica --split test --img_wh 320 240 --N_importance 64 --chunk 40000 --ckpt_path ckpts/debug_replica/

python extract_color_mesh.py --root_dir chair/ --dataset_name blender --scene_name chair --img_wh 400 400 --ckpt_path ckpts/debug-0607/\{epoch\:d\}/epoch\=30-step\=154.ckpt  --sigma_threshold 20.0 --N_grid 256

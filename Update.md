代码更新
-------------------------------------
### 2022-03-09

主要改动: celoss 因为类别多, 很容易就优化不动. 需要合并 label. 使用了@zijian 编辑过的label.



### 2022-03-08

1. 加入了人脸数据的处理, `datasets/llff_cls.py`. 

   主要改动:train / val 加入了face-parsing的结果, **目前使用的是没有编辑过的结果**.

2. render 的处理, `models/rendering.py`, 

    主要改动: 新增了 `render_rays_3d`. 对每个ray输出所属类别, shape: (N_rays, num_cls). 并在cross entropy中使用parse的输出监督render结果 (见 `losses.py`， weights采用默认的可以正常训练).

3. Model 的修改, `models/nerf_cls.py`.

    主要改动: 新增了MLP 输出 xyz 的 所属类别.

在以上改动后可正常训练 / 训练(~~psnr并不随着训练连的进行而增加~~), 

```python
# train
 DEBUG=False python train.py 
   --dataset_name llff_cls \
   --root_dir bowen_tou \
   --N_importance 64 --img_wh 1920 1080 \
   --num_epochs 30 --batch_size 1024 \
   --optimizer adam --lr 5e-4 \
   --lr_scheduler steplr --decay_step 10 20 --decay_gamma 0.5 \
   --exp_name exp-bowen --d3 --loss_type msece --spheric
# val
export CKPT=./ckpts/exp-bowen/epoch=2.ckpt
python eval.py \
   --root_dir bowen_tou  \
   --dataset_name llff_cls --scene_name bowen \
   --img_wh 1920 1080 --N_importance 64 --ckpt_path $CKPT  --d3 \
   --spheric
```
**记得加上 `--spheric`  因为是3d环绕采集的**


*一些并不重要的改动*

a. 可视化csv的曲线:  `python vis_log.py $CSV$ $KEY$`
   
   会在原目录下保存对应key的变化曲线.


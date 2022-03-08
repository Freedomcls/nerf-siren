代码更新
-------------------------------------

### 2022-03-08

1. 加入了人脸数据的处理, `datasets/llff_cls.py`. 
   主要改动:train / val 加入了face-parsing的结果, **目前使用的是没有编辑过的结果**.

2. render 的处理, `models/rendering.py`, 
    主要改动: 新增了 `render_rays_3d`. 对每个ray输出所属类别, shape: (N_rays, num_cls). 并在cross entropy中使用parse的输出监督render结果 (见 `losses.py`).

3. Model 的修改, `models/nerf_cls.py`.
    主要改动: 新增了MLP 输出 xyz 的 所属类别.

在以上改动后可正常训练(但不收敛, psnr并不随着训练连的进行而增加), 测试待测.

```python
 DEBUG=False python train.py 
   --dataset_name llff_cls \
   --root_dir bowen_tou \
   --N_importance 64 --img_wh 1920 1080 \
   --num_epochs 30 --batch_size 1024 \
   --optimizer adam --lr 5e-4 \
   --lr_scheduler steplr --decay_step 10 20 --decay_gamma 0.5 \
   --exp_name exp-bowen --d3 --loss_type msece
```

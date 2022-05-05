import os, sys
from opt import get_opts
import torch
# system
from system import NeRFSystem, NeRF3DSystem, NeRF3DSystem_ib
# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.plugins import DDPPlugin


if __name__ == '__main__':
    hparams = get_opts()
    if hparams.mode == 'd3':
        print("Use NeRF_3D")
        system = NeRF3DSystem(hparams)
    elif hparams.mode == "d3_ib":
        print("Use NeRF_3D Img Batch")
        system =  NeRF3DSystem_ib(hparams)
    else:
        system = NeRFSystem(hparams)

    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(f'ckpts/{hparams.exp_name}',
                                                                '{epoch:d}'),
                                          monitor='val/loss',
                                          mode='min',
                                          save_top_k=10,)

    logger = TestTubeLogger(
        save_dir="logs",
        name=hparams.exp_name,
        debug=False,
        create_git_tag=False
    )

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      checkpoint_callback=checkpoint_callback,
                      resume_from_checkpoint=hparams.ckpt_path,
                      logger=logger,
                      weights_summary=None,
                      progress_bar_refresh_rate=1,
                      gpus=hparams.num_gpus,
                      distributed_backend='ddp' if hparams.num_gpus>1 else None,
                      plugins=DDPPlugin(find_unused_parameters=True),
                      num_sanity_val_steps=1,
                      benchmark=True,
                      profiler=hparams.num_gpus==1)

    trainer.fit(system)

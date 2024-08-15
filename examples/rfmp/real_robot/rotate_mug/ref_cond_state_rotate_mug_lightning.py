import os

import matplotlib
matplotlib.use('TkAgg')
from omegaconf import DictConfig, OmegaConf
from glob import glob
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

import sys
sys.path.append('/home/dia1rng/hackathon/flow-matching-policies/manifm')
from model_trajectories_rotate_mug import ManifoldStateTrajectoriesMugRotateFMLitModule
sys.path.append('/home/dia1rng/hackathon/flow-matching-policies/data/real_robot')
from dummy_robot_arm1 import get_loaders
from types import SimpleNamespace

import argparse


if __name__ == '__main__':
    # the name of dataset: dish_grasp
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='tMLP', type=str)
    args = parser.parse_args()

    print('load args')
    # Load config
    cfg = OmegaConf.load('refcond_rfm_rotate_mug.yaml')
    cfg.model_type = 'Unet'
    data_folder = cfg.data_dir
    data_args = SimpleNamespace()
    data_args.ablation = 'vf_vg'
    data_args.num_stack = 2
    data_args.frameskip = 1
    data_args.no_crop = False
    data_args.crop_percent = 0.1
    data_args.resized_height_v = cfg.image_height
    data_args.resized_width_v = cfg.image_width
    data_args.len_lb = cfg.n_pred - 1
    data_args.sampling_time = 250
    data_args.norm = False
    data_args.source = True
    # Load dataset
    train_loader, val_loader, _ = get_loaders(batch_size=cfg.optim.batch_size, args=data_args, data_folder=data_folder,
                                                    drop_last=False)
    print('data load')
    # Construct model
    model = ManifoldStateTrajectoriesMugRotateFMLitModule(cfg)
    print(model)
    add_info = '_tttt'
    # Checkpointing, logging, and other misc.
    checkpoints_dir = "checkpoints/checkpoints_rfm_" + cfg.data + \
                      "_n" + str(cfg.n_pred) + "_r" + str(cfg.n_ref) + "_c" + str(cfg.n_cond) + "_w" + str(
        cfg.w_cond) + cfg.model_type + add_info

    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoints_dir,
            monitor="val/loss_best",
            mode="min",
            filename="epoch-{epoch:03d}_step-{global_step}" + cfg.model_type + add_info,
            auto_insert_metric_name=False,
            save_top_k=1,
            save_last=True,
            every_n_train_steps=cfg.get("ckpt_every", None),
        ),
        LearningRateMonitor(),
    ]

    slurm_plugin = pl.plugins.environments.SLURMEnvironment(auto_requeue=False)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["cwd"] = os.getcwd()
    loggers = [pl.loggers.CSVLogger(save_dir=".")]
    if cfg.use_wandb:
        loggers.append(
            pl.loggers.WandbLogger(
                save_dir=".",
                name=f"{cfg.data}_" + cfg.model_type,
                project="RFMP_rotate_mug",
                log_model=False,
                config=cfg_dict,
                resume=True,
            )
        )
    trainer = pl.Trainer(
        max_steps=cfg.optim.num_iterations,
        accelerator="gpu",
        devices=1,
        logger=loggers,
        val_check_interval=cfg.val_every,
        check_val_every_n_epoch=5,
        callbacks=callbacks,
        precision=cfg.get("precision", 32),
        gradient_clip_val=cfg.optim.grad_clip,
        plugins=slurm_plugin if slurm_plugin.detect() else None,
        num_sanity_val_steps=0,
        max_epochs=200,
    )

    # If we specified a checkpoint to resume from, use it
    checkpoint = cfg.get("resume", None)

    # Check if a checkpoint exists in this working directory.  If so, then we are resuming from a pre-emption
    # This takes precedence over a command line specified checkpoint
    checkpoints = glob(checkpoints_dir + "/**/*.ckpt", recursive=True)
    if len(checkpoints) > 0:
        # Use the checkpoint with the latest modification time
        checkpoint = sorted(checkpoints, key=os.path.getmtime)[-1]

    trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint)
    trainer.save_checkpoint(checkpoints_dir + '/model.ckpt')
    train_metrics = trainer.callback_metrics

    print(train_metrics)

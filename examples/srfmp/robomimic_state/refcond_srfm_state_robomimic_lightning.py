import os

from omegaconf import OmegaConf
from glob import glob
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

from stable_flow.stable_model_trajs_robomimic_pl_learntau import SRFMRobomimicLTModule
import wandb
from data.robomimic.get_dataloadar import get_robomimic_dataloadar


'''
train SRFMP on robomimic tasks with state-based observation

change 'task' in 'refcond_srfm_robomimic.yaml' to train different robomimic tasks
'''


if __name__ == '__main__':
    # Load config
    cfg = OmegaConf.load('refcond_srfm_robomimic.yaml')
    cfg.model_type = 'Unet'
    # Load dataset
    train_loader, val_loader = get_robomimic_dataloadar(cfg)
    print('data load')
    if cfg.use_wandb:
        # remember to change to your own wandb_key
        wb_key = OmegaConf.load('/home/dia1rng/hackathon/flow-matching-policies/wandb_key.yml')
        wandb.login(key=wb_key.wandb_key, relogin=True)

    # set checkpoints
    save_gap = 20
    save_num = 5
    total_epoch = save_gap * save_num
    add_info = '_saveevery' + str(save_gap) + '_total' + str(total_epoch)

    # Construct model
    model = SRFMRobomimicLTModule(cfg)
    print(model)

    # Checkpointing, logging, and other misc.
    checkpoints_dir = "checkpoints/checkpoints_rfm_" + cfg.data + \
                      "_n" + str(cfg.n_pred) + "_r" + str(cfg.n_ref) + "_c" + str(cfg.n_cond) + "_w" + str(
        cfg.w_cond) + cfg.model_type + cfg.task + add_info

    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoints_dir,
            filename="epoch-{epoch:03d}" + cfg.model_type + cfg.task,
            save_top_k=-1,
            save_last=True,
            every_n_epochs=save_gap,
        ),
    ]

    slurm_plugin = pl.plugins.environments.SLURMEnvironment(auto_requeue=False)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["cwd"] = os.getcwd()
    loggers = [pl.loggers.CSVLogger(save_dir=".")]
    if cfg.use_wandb:
        loggers.append(
            pl.loggers.WandbLogger(
                save_dir=".",
                name=f"{cfg.data}_" + cfg.model_type + cfg.task,
                project="SRFMP_Robomimic",
                log_model=False,
                config=cfg_dict,
                resume=False,
            )
        )
    trainer = pl.Trainer(
        max_steps=cfg.optim.num_iterations,
        accelerator="gpu",
        devices=1,
        logger=loggers,
        val_check_interval=cfg.val_every,
        check_val_every_n_epoch=save_gap,
        callbacks=callbacks,
        precision=cfg.get("precision", 32),
        gradient_clip_val=cfg.optim.grad_clip,
        plugins=slurm_plugin if slurm_plugin.detect() else None,
        num_sanity_val_steps=0,
        max_epochs=total_epoch,
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
    train_metrics = trainer.callback_metrics

    print(train_metrics)

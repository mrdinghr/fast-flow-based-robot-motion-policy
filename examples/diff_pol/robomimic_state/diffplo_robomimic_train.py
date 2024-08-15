import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel
from tqdm.auto import tqdm
import sys

sys.path.append('/home/dia1rng/hackathon/flow-matching-policies/data/robomimic')
from get_dataloadar import get_robomimic_dataloadar
from manifm.datasets import get_manifold
from omegaconf import OmegaConf
import os
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D


class DiffpolRobomimicDataUtils:
    def __init__(self, cfg, device=torch.device('cuda')):
        self.cfg = cfg
        self.manifold, self.dim = get_manifold(cfg)
        self.device = device
        self.n_ref = cfg.n_ref
        self.n_pred = cfg.n_pred
        self.output_dim = self.dim * self.n_pred
        if cfg.task == 'can':
            self.ref_object_feature = 14
        elif cfg.task == 'lift':
            self.ref_object_feature = 10
        elif cfg.task == 'square':
            self.ref_object_feature = 14
        elif cfg.task == 'tool_hang':
            self.ref_object_feature = 44
        else:
            assert False, 'Wrong Task setting'
        self.feature_dim = self.ref_object_feature + 3 + 4 + 2

    def unpack_predictions_reference_conditioning_samples(self, batch: torch.Tensor):
        B = batch['actions'].shape[0]
        x1 = batch['actions'][:, self.cfg.n_ref - 1:, :].to(
            self.device)  # shape B * n_pred * 7
        xref_object = batch['obs']['object'][:, :self.cfg.n_ref, :]  # shape B * n_ref * 14
        xref_robo_pos = batch['obs']['robot0_eef_pos'][:, :self.cfg.n_ref, :]
        xref_robo_quat = batch['obs']['robot0_eef_quat'][:, :self.cfg.n_ref, :]
        xref_robo_grip = batch['obs']['robot0_gripper_qpos'][:, :self.cfg.n_ref, :]
        xref = torch.cat((xref_object, xref_robo_pos, xref_robo_quat, xref_robo_grip), dim=-1).reshape(
            (B, self.n_ref * self.feature_dim)).float().to(self.device)
        return x1, xref


def train(nets, cfg, noise_scheduler, train_loader, val_loader, data_utils, num_epochs=300,
          device=torch.device('cuda')):
    ema = EMAModel(nets)
    optimizer = torch.optim.AdamW(
        params=nets.parameters(),
        lr=cfg.optim.lr, weight_decay=cfg.optim.wd)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.optim.num_iterations,
    )

    # lr_scheduler = get_scheduler(
    #     name='cosine',
    #     optimizer=optimizer,
    #     num_warmup_steps=500,
    #     num_training_steps=len(train_loader) * num_epochs
    # )
    cur_best_val_loss = None
    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        epoch_total_loss = list()
        total_val_loss = list()
        for epoch_idx in tglobal:
            epoch_loss = list()
            with tqdm(train_loader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    # data normalized in dataset
                    # device transfer
                    naction, obs_cond = data_utils.unpack_predictions_reference_conditioning_samples(batch=nbatch)
                    B = naction.shape[0]
                    # sample noise to add to actions
                    noise = torch.randn(naction.shape, device=device)

                    # sample a diffusion iteration for each data point
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (B,), device=device
                    ).long()

                    # add noise to the clean images according to the noise magnitude at each diffusion iteration
                    # (this is the forward diffusion process)
                    noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)

                    # predict the noise residual
                    noise_pred = noise_pred_net(
                        noisy_actions, timesteps, global_cond=obs_cond)

                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    # update Exponential Moving Average of the model weights
                    ema.step(nets.parameters())

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
            tglobal.set_postfix(loss=np.mean(epoch_loss))
            epoch_total_loss.append(np.mean(epoch_loss))
            if epoch_idx in [30, 50, 75, 100, 150, 200, 250, 300]:
                check_point(nets, cfg.task + '/' + cfg.diffusion_model + '_epoch' + str(epoch_idx) + '.pt')
            if epoch_idx % 5 == 0:
                with tqdm(val_loader, desc='Val_Batch', leave=False) as valepoch:
                    val_loss_list = []
                    with torch.no_grad():
                        for valbatch in valepoch:
                            # data normalized in dataset
                            # device transfer
                            val_naction, obs_cond = data_utils.unpack_predictions_reference_conditioning_samples(
                                batch=valbatch)
                            B = val_naction.shape[0]
                            # sample noise to add to actions
                            val_noise = torch.randn(val_naction.shape, device=device)

                            # sample a diffusion iteration for each data point
                            val_timesteps = torch.randint(
                                0, noise_scheduler.config.num_train_timesteps,
                                (B,), device=device
                            ).long()

                            # add noise to the clean images according to the noise magnitude at each diffusion iteration
                            # (this is the forward diffusion process)
                            val_noisy_actions = noise_scheduler.add_noise(
                                val_naction, val_noise, val_timesteps)

                            # predict the noise residual
                            val_noise_pred = noise_pred_net(
                                val_noisy_actions, val_timesteps, global_cond=obs_cond)

                            # L2 loss
                            val_loss = nn.functional.mse_loss(val_noise_pred, val_noise)
                            val_loss_list.append(val_loss.cpu().numpy())

                if cur_best_val_loss is None:
                    cur_best_val_loss = np.mean(val_loss_list)
                    check_point(nets, cfg.task + '/' + cfg.diffusion_model + '_bestval.pt')
                    np.savetxt(cfg.task + '/' + cfg.diffusion_model + 'best_val_epoch.txt', [epoch_idx])
                    total_val_loss.append(cur_best_val_loss)
                else:
                    if cur_best_val_loss > np.mean(val_loss_list):
                        check_point(nets, cfg.task + '/' + cfg.diffusion_model + '_bestval.pt')
                        np.savetxt(cfg.task + '/' + cfg.diffusion_model + 'best_val_epoch.txt', [epoch_idx])
                        cur_best_val_loss = np.mean(val_loss_list)
                        total_val_loss.append(cur_best_val_loss)
            np.save(cfg.task + '/' + cfg.diffusion_model + 'epoch_loss_list.npy', epoch_total_loss)
            np.save(cfg.task + '/' + cfg.diffusion_model + 'val_apoch_loss_list.npy', total_val_loss)


def check_point(model, filename):
    torch.save(model.state_dict(), filename)


if __name__ == '__main__':
    cfg = OmegaConf.load('diffpol_robomimic.yaml')
    print('task ', cfg.task)
    print('diffusion model ', cfg.diffusion_model)
    if not os.path.exists(cfg.task):
        os.mkdir(cfg.task)
    n_ref = cfg.n_ref
    train_loader, val_loader = get_robomimic_dataloadar(cfg)
    num_epochs = 300
    device = torch.device('cuda')
    data_utils = DiffpolRobomimicDataUtils(cfg, device=device)
    noise_pred_net = ConditionalUnet1D(
        input_dim=data_utils.dim,
        global_cond_dim=n_ref * data_utils.feature_dim,
        down_dims=[256, 512, 1024],
    )
    if cfg.diffusion_model == 'DDPM':
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule='squaredcos_cap_v2',
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type='epsilon'
        )
    elif cfg.diffusion_model == 'DDIM':
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=100,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule='squaredcos_cap_v2',
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type='epsilon'
        )
    else:
        assert False, 'Wrong Diffusion Model Type'
    _ = noise_pred_net.to(device)
    train(noise_pred_net, noise_scheduler=noise_scheduler, cfg=cfg, train_loader=train_loader, val_loader=val_loader,
          num_epochs=num_epochs, data_utils=data_utils)

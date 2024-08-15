import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel
from tqdm.auto import tqdm
import sys

sys.path.append('/home/dia1rng/hackathon/flow-matching-policies/data/robomimic')
from get_image_dataloadar import get_robomimic_image_dataloadar
from manifm.datasets import get_manifold
from omegaconf import OmegaConf
import os
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from manifm.vision.resnet_models import get_resnet, replace_bn_with_gn
import torchvision.transforms as Transform


class DiffpolRobomimicDataUtils:
    def __init__(self, cfg, nets, device=torch.device('cuda')):
        self.cfg = cfg
        self.manifold, self.dim = get_manifold(cfg)
        self.device = device
        self.n_ref = cfg.n_ref
        self.n_pred = cfg.n_pred
        self.output_dim = self.dim * self.n_pred
        self.img_feature_dim = 512
        self.feature_dim = self.img_feature_dim * 2 + 3 + 4 + 2
        self.nets = nets
        if cfg.task == 'tool_hang':
            self.crop_transform = Transform.RandomCrop((216, 216))
        else:
            self.crop_transform = Transform.RandomCrop((76, 76))

    def image_to_features(self, x):
        x = self.crop_transform(x)
        image_feaeture = self.nets['vision_encoder'](x.flatten(end_dim=1))
        return image_feaeture.reshape(*x.shape[:2], -1)

    def grip_image_to_features(self, x):
        x = self.crop_transform(x)
        image_feaeture = self.nets['grip_vision_encoder'](x.flatten(end_dim=1))
        return image_feaeture.reshape(*x.shape[:2], -1)

    def unpack_predictions_reference_conditioning_samples(self, batch: torch.Tensor):
        B = batch['actions'].shape[0]
        x1 = batch['actions'][:, self.cfg.n_ref - 1:, :].to(
            self.device)  # shape B * n_pred * 7
        xref_image = batch['obs']['agentview_image'][:, :self.cfg.n_ref, :]
        xref_image = xref_image.moveaxis(-1, -3).float()
        xref_image_scalar = self.image_to_features(xref_image)
        xref_grip_image = batch['obs']['robot0_eye_in_hand_image'][:, :self.cfg.n_ref, :]
        xref_grip_image = xref_grip_image.moveaxis(-1, -3).float()
        xref_grip_iamge_scalar = self.grip_image_to_features(xref_grip_image)

        xref_robo_pos = batch['obs']['robot0_eef_pos'][:, :self.cfg.n_ref, :]
        xref_robo_quat = batch['obs']['robot0_eef_quat'][:, :self.cfg.n_ref, :]
        xref_robo_grip = batch['obs']['robot0_gripper_qpos'][:, :self.cfg.n_ref, :]
        xref = torch.cat((xref_image_scalar, xref_grip_iamge_scalar, xref_robo_pos, xref_robo_quat, xref_robo_grip), dim=-1).reshape(
            (B, self.n_ref * self.feature_dim)).float().to(self.device)
        return x1, xref


def train(nets, cfg, noise_scheduler, train_loader, val_loader, data_utils, num_epochs=300,
          device=torch.device('cuda'), save_gap=10):
    ema = EMAModel(nets)
    num_epochs = save_gap * 5
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
                    noise_pred = nets['noise_pred_net'](
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
            if (epoch_idx + 1) % save_gap == 0:
                check_point(nets, cfg.task + '/' + cfg.diffusion_model + '_epoch' + str(epoch_idx) + '.pt')
            np.save(cfg.task + '/' + cfg.diffusion_model + 'epoch_loss_list.npy', epoch_total_loss)


def check_point(model, filename):
    torch.save(model.state_dict(), filename)


if __name__ == '__main__':
    cfg = OmegaConf.load('diffpol_robomimic_img.yaml')
    print('task ', cfg.task)
    print('diffusion model ', cfg.diffusion_model)
    if not os.path.exists(cfg.task):
        os.mkdir(cfg.task)
    n_ref = cfg.n_ref
    train_loader, val_loader = get_robomimic_image_dataloadar(cfg)
    num_epochs = 300
    device = torch.device('cuda')

    noise_pred_net = ConditionalUnet1D(
        input_dim=7,
        global_cond_dim=n_ref * (512 * 2 + 3 + 4 + 2),
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
    vision_encoder = get_resnet('resnet18')
    vision_encoder = replace_bn_with_gn(vision_encoder)
    grip_vision_encoder = get_resnet('resnet18')
    grip_vision_encoder = replace_bn_with_gn(grip_vision_encoder)
    nets = torch.nn.ModuleDict({'vision_encoder': vision_encoder, 'noise_pred_net': noise_pred_net,
                                'grip_vision_encoder': grip_vision_encoder})
    data_utils = DiffpolRobomimicDataUtils(cfg, nets=nets, device=device)
    if cfg.task == 'tool_hang':
        save_gap = 40
    else:
        save_gap = 20
    train(nets, noise_scheduler=noise_scheduler, cfg=cfg, train_loader=train_loader, val_loader=val_loader,
          num_epochs=num_epochs, data_utils=data_utils, save_gap=save_gap)

import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel
from tqdm.auto import tqdm
from manifm.datasets import get_loaders
from omegaconf import OmegaConf
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from manifm.vision.resnet_models import get_resnet, replace_bn_with_gn
from diffusers.optimization import get_scheduler
from manifm.manifolds import Sphere, ProductManifoldTrajectories


# seq_unpack: True: img1, pos1, img2, pos2 False: img1, img2, pos1, pos2
def train(nets, cfg, train_loader, val_loader, num_epochs=300, seq_unpack=True, sphere_prior_sample=True):
    manifold = [(Sphere(), 3) for _ in range(cfg.n_pred)]
    manifold = ProductManifoldTrajectories(*manifold)
    ema = EMAModel(nets)
    optimizer = torch.optim.AdamW(
        params=nets.parameters(),
        lr=cfg.optim.lr, weight_decay=cfg.optim.wd)

    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(train_loader) * num_epochs
    )
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
                    nimage = nbatch['obs']['image'].float().to(device)
                    nagent_pos = nbatch['obs']['agent_pos'].float().to(device)
                    naction = nbatch['action'].float().to(device)
                    B = nagent_pos.shape[0]

                    # encoder vision features
                    image_features = nets['vision_encoder'](
                        nimage.flatten(end_dim=1))
                    image_features = image_features.reshape(
                        *nimage.shape[:2], -1)
                    # (B,obs_horizon,D)

                    # concatenate vision feature and low-dim obs
                    if seq_unpack:
                        obs_features = torch.cat([image_features, nagent_pos], dim=-1)
                    else:
                        nagent_pos = nagent_pos.reshape((B, cfg.n_ref * 3))
                        image_features = image_features.reshape((B, cfg.n_ref * 512))
                        obs_features = torch.cat([image_features, nagent_pos], dim=-1)
                    obs_cond = obs_features.flatten(start_dim=1)
                    # (B, obs_horizon * obs_dim)

                    # sample noise to add to actions
                    if not sphere_prior_sample:
                        noise = torch.randn(naction.shape, device=device)
                    else:
                        noise = manifold.random_base(B, cfg.n_pred * 3).reshape((B, cfg.n_pred, 3))

                    # sample a diffusion iteration for each data point
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (B,), device=device
                    ).long()

                    # add noise to the clean images according to the noise magnitude at each diffusion iteration
                    # (this is the forward diffusion process)
                    noisy_actions = noise_scheduler.add_noise(
                        naction, noise, timesteps)

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
                    check_point(nets, 'model.pt')
            tglobal.set_postfix(loss=np.mean(epoch_loss))
            epoch_total_loss.append(np.mean(epoch_loss))
            if epoch_idx % 5 == 0:
                with tqdm(val_loader, desc='Val_Batch', leave=False) as valepoch:
                    val_loss_list = []
                    with torch.no_grad():
                        for valbatch in valepoch:
                            valnimage = valbatch['obs']['image'].float().to(device)
                            valnagent_pos = valbatch['obs']['agent_pos'].float().to(device)
                            valnaction = valbatch['action'].float().to(device)
                            B = valnagent_pos.shape[0]
                            val_image_features = nets['vision_encoder'](valnimage.flatten(end_dim=1))
                            val_image_features = val_image_features.reshape(*valnimage.shape[:2], -1)
                            if seq_unpack:
                                valobs_features = torch.cat([val_image_features, valnagent_pos], dim=-1)
                            else:
                                val_image_features = val_image_features.reshape((B, cfg.n_ref * 512))
                                valnagent_pos = valnagent_pos.reshape((B, cfg.n_ref * 3))
                                valobs_features = torch.cat([val_image_features, valnagent_pos], dim=-1)
                            valobs_cond = valobs_features.flatten(start_dim=1)
                            if not sphere_prior_sample:
                                valnoise = torch.randn(valnaction.shape, device=device)
                            else:
                                valnoise = manifold.random_base(B, cfg.n_pred * 3).reshape((B, cfg.n_pred, 3))
                            valtimesteps = torch.randint(
                                0, noise_scheduler.config.num_train_timesteps,
                                (B,), device=device
                            ).long()
                            valnoisy_actions = noise_scheduler.add_noise(
                                valnaction, valnoise, valtimesteps)
                            valnoise_pred = noise_pred_net(valnoisy_actions, valtimesteps, global_cond=valobs_cond)
                            val_loss = nn.functional.mse_loss(valnoise_pred, valnoise)
                            val_loss_list.append(val_loss.cpu().numpy())

                if cur_best_val_loss is None:
                    cur_best_val_loss = np.mean(val_loss_list)
                    check_point(nets, cfg.diffusion_model + '.pt')
                    np.savetxt(cfg.diffusion_model + 'best_val_epoch.txt', [epoch_idx])
                    total_val_loss.append(cur_best_val_loss)
                else:
                    if cur_best_val_loss > np.mean(val_loss_list):
                        check_point(nets, cfg.diffusion_model + '.pt')
                        np.savetxt(cfg.diffusion_model + 'best_val_epoch.txt', [epoch_idx])
                        cur_best_val_loss = np.mean(val_loss_list)
                        total_val_loss.append(cur_best_val_loss)
            np.save(cfg.diffusion_model + 'epoch_loss_list.npy', epoch_total_loss)
            np.save(cfg.diffusion_model + 'val_apoch_loss_list.npy', total_val_loss)


def check_point(model, filename):
    torch.save(model.state_dict(), filename)


if __name__ == '__main__':
    cfg = OmegaConf.load('diffpol_sphere_vision_pusht.yaml')
    train_loader, val_loader, test_loader = get_loaders(cfg)
    num_epochs = 500
    action_dim = 3
    obs_dim = 512 + action_dim
    obs_horizon = 2
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * obs_horizon,
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
    device = torch.device('cuda')
    _ = noise_pred_net.to(device)
    vision_encoder = get_resnet('resnet18')
    vision_encoder = replace_bn_with_gn(vision_encoder)
    _ = vision_encoder.to(device)
    nets = torch.nn.ModuleDict({'vision_encoder': vision_encoder, 'noise_pred_net': noise_pred_net})
    train(nets, cfg=cfg, train_loader=train_loader, val_loader=val_loader, num_epochs=num_epochs, seq_unpack=False)


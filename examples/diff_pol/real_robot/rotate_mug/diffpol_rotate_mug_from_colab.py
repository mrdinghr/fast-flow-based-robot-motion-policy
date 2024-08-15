import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import sys
sys.path.append('/home/dia1rng/hackathon/flow-matching-policies/data/real_robot')
from vision_audio_robot_arm import get_loaders
from omegaconf import OmegaConf
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from manifm.vision.resnet_models import get_resnet, replace_bn_with_gn
from types import SimpleNamespace


class RotateMugDataUtils:
    def __init__(self, cfg, nets, device=torch.device('cuda')):
        self.cfg = cfg
        self.device = device
        # self.pos_quat_max = torch.tensor([0.7, 0.3, 0.4, 1., 1., 1., 1.]).to(self.device)
        # self.pos_quat_min = torch.tensor([0.2, -0.1, 0., -1., -1., -1., -1.]).to(self.device)
        # new simple dataset
        self.pos_quat_max = torch.tensor([0.6, 0.25, 0.4, 1., 1., 1., 1.]).to(self.device)
        self.pos_quat_min = torch.tensor([0.2, 0., 0.2, -1., -1., -1., -1.]).to(self.device)

        self.dim = 3 + 4 + 1
        self.normalize = cfg.normalize_pos_quat

        self.n_ref = cfg.n_ref
        self.n_pred = cfg.n_pred
        self.nets = nets

    def normalize_pos_quat(self, pos_quat):
        return 2 * (pos_quat - self.pos_quat_min.to(self.device)) / (
                self.pos_quat_max.to(self.device) - self.pos_quat_min.to(self.device)) - 1

    def denormalize_pos_quat(self, norm_pos_quat):
        return (norm_pos_quat + 1) / 2 * (
                self.pos_quat_max.to(self.device) - self.pos_quat_min.to(self.device)) + self.pos_quat_min.to(
            self.device)

    def normalize_grip(self, grip_state):
        return grip_state / 0.03

    def denormalize_grip(self, norm_grip_state):
        return norm_grip_state * 0.03

    def image_to_features(self, x):
        image_feaeture = self.nets['vision_encoder'](x.flatten(end_dim=1))
        image_feaeture = image_feaeture.reshape(*x.shape[:2], -1)
        return image_feaeture.flatten(start_dim=1)

    def unpack_predictions_reference_conditioning_samples(self, batch: torch.Tensor):
        # Unpack batch vector into different components
        # Assumes greyscale images
        # if self.del_unmove_batch:
        # batch = self.check_batch_no_move(batch)
        B = batch['observation']['v_fix'].shape[0]

        x1_pos_quat = batch['traj']['target_pos_quat']['action'].to(self.device)
        xref_pos_quat = batch['traj']['target_pos_quat']['obs'].to(self.device).float()
        xref_gripper = batch['traj']['gripper']['obs'].to(self.device)
        # make sure x1[0] == xref[-1]
        x1_pos_quat = torch.cat([xref_pos_quat[:, -1:, :], x1_pos_quat], dim=1)
        # x1_pos_quat

        # regulate the quaternion, because like  0 1 0 0 and 0 -1 0 0 are same rotation. this line is to make sure all rotation in same way
        x1_pos_quat[..., 3:] *= torch.sign(x1_pos_quat[..., 4]).unsqueeze(-1)
        x1_gripper = batch['traj']['gripper']['action'].unsqueeze(-1).to(self.device)
        x1_gripper = torch.cat([xref_gripper[..., -1].unsqueeze(-1).unsqueeze(-1), x1_gripper], dim=1)
        if self.normalize:
            x1_pos_quat = self.normalize_pos_quat(x1_pos_quat)
        x1 = torch.cat((x1_pos_quat, x1_gripper), dim=2).float()

        # image reference scalar: fix_img
        fix_img = batch['observation']['v_fix'].to(self.device)  # shape B * n_ref * 3 * H * W
        fix_img_feature = self.image_to_features(fix_img)
        img_scalar = fix_img_feature.reshape((B, self.n_ref * 512))

        xref_pos_quat[..., 3:] *= torch.sign(xref_pos_quat[..., 4]).unsqueeze(-1)
        if self.normalize:
            xref_pos_quat = self.normalize_pos_quat(xref_pos_quat)
        xref_pos_quat = xref_pos_quat.reshape((B, self.n_ref * (self.dim - 1)))
        xref = torch.cat((img_scalar, xref_pos_quat, xref_gripper), dim=1).float()
        return x1, xref


def train(nets, cfg, train_loader, val_loader, num_epochs=300):
    ema = EMAModel(nets)
    data_utils = RotateMugDataUtils(cfg, nets=nets, device=device)
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
            if epoch_idx % 5 == 0:
                with tqdm(val_loader, desc='Val_Batch', leave=False) as valepoch:
                    val_loss_list = []
                    with torch.no_grad():
                        for valbatch in valepoch:
                            # data normalized in dataset
                            # device transfer
                            val_naction, obs_cond = data_utils.unpack_predictions_reference_conditioning_samples(batch=valbatch)
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
    cfg = OmegaConf.load('diffpol_rotatemug.yaml')
    n_ref = cfg.n_ref

    vision_feature_dim = 512 + 8

    data_folder = cfg.data_dir
    data_args = SimpleNamespace()
    data_args.ablation = 'vf_vg'
    data_args.num_stack = 2
    data_args.frameskip = 1
    data_args.no_crop = False
    data_args.crop_percent = 0.2
    data_args.resized_height_v = cfg.image_height
    data_args.resized_width_v = cfg.image_width
    data_args.len_lb = cfg.n_pred - 1
    data_args.sampling_time = 250
    data_args.source = False
    data_args.catg = "resample"
    data_args.norm_type = "limit"
    data_args.smooth_factor = (1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5)
    # Load dataset
    train_loader, val_loader, _ = get_loaders(batch_size=cfg.optim.batch_size, args=data_args, data_folder=data_folder,
                                              drop_last=False, debug=True, val_batch_size=cfg.optim.val_batch_size)

    num_epochs = 300
    action_dim = 3 + 4 + 1

    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=n_ref * vision_feature_dim,
        down_dims=[128, 256, 512],
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
    train(nets, cfg=cfg, train_loader=train_loader, val_loader=val_loader, num_epochs=num_epochs)


import matplotlib.pyplot as plt
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from manifm.vision.resnet_models import get_resnet, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from omegaconf import OmegaConf
from glob import glob
import sys
sys.path.append('/home/dia1rng/hackathon/flow-matching-policies/data/real_robot')
from vision_audio_robot_arm import get_loaders
from types import SimpleNamespace
import time
from tqdm import tqdm
import numpy as np
import torch
import cv2
from diffpol_rotate_mug_from_colab import RotateMugDataUtils
import matplotlib
matplotlib.use('TkAgg')


# todo
def compare_demo_gen(batch, action):
    demo_pos_quat = batch['future_pos_quat'][0]
    demo_grip_pos = batch['future_gripper'][0]
    gen_pos_quat = action[:, :7]
    gen_grip_pos = action[:, -1]
    plt.figure(figsize=(16, 16))
    for i in range(7):
        plt.subplot(8, 1, i + 1)
        plt.plot(demo_pos_quat[:, i], c='r', alpha=0.2, label='demo')
        plt.plot(gen_pos_quat[:, i], c='b', alpha=0.2, label='gen')
    plt.legend()
    plt.show()


def infer_whole_traj(data_loadar, nets, data_utils, noise_scheduler, action_horizon=8, save_video=False, no_crop=True,
                     num_diffusion_iters=10, execute_horizon=8):
    gen_state_list = []
    demo_state_list = []
    demo_grip_list = []
    fix_img_list = []
    if save_video:
        fps = 3
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        width = 640
        height = 480
        video_writer = cv2.VideoWriter(checkpoints_dir + '/loadar_img.mp4', fourcc, fps, (width, height))
    B = 1
    for i, batch in tqdm(enumerate(data_loadar)):
        if i % action_horizon == 0:
            target_action, obs_cond = data_utils.unpack_predictions_reference_conditioning_samples(batch)
            noise_scheduler.set_timesteps(num_diffusion_iters)
            noisy_action = torch.randn(
                (B, 16, action_dim), device=device)
            naction = noisy_action
            start_time = time.time()
            with torch.no_grad():
                for k in noise_scheduler.timesteps:
                    # predict noise
                    noise_pred = nets['noise_pred_net'](
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )

                    # inverse diffusion step (remove noise)
                    naction = noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample
            naction = naction[0]
            if cfg.normalize_pos_quat:
                naction[..., :-1] = data_utils.denormalize_pos_quat(naction[..., :-1])
                # naction[..., -1] = data_utils.denormalize_grip(naction[..., -1])
            print('infer uses time ' + str(time.time() - start_time))
            if i == 0:
                gen_state_list = naction[:execute_horizon].cpu().numpy()
                regulate_quat_sign = torch.sign(
                    batch['traj']['target_pos_quat']['action'][0][:execute_horizon][..., 4]).unsqueeze(-1)
                demo_pos_quat = batch['traj']['target_pos_quat']['action'][0][:execute_horizon]
                demo_pos_quat[..., 3:] *= regulate_quat_sign
                demo_state_list = demo_pos_quat.cpu().numpy()
                demo_grip_list = batch['traj']['gripper']['action'][0][:execute_horizon].cpu().numpy()
            else:
                gen_state_list = np.concatenate([gen_state_list, naction[:execute_horizon].cpu().numpy()])
                regulate_quat_sign = torch.sign(
                    batch['traj']['target_pos_quat']['action'][0][:execute_horizon][..., 4]).unsqueeze(-1)
                demo_pos_quat = batch['traj']['target_pos_quat']['action'][0][:execute_horizon]
                demo_pos_quat[..., 3:] *= regulate_quat_sign
                demo_state_list = np.concatenate([demo_state_list, demo_pos_quat.cpu().numpy()])
                demo_grip_list = np.concatenate(
                    [demo_grip_list, batch['traj']['gripper']['action'][0][:execute_horizon].cpu().numpy()])
        obs = batch["observation"]
        image_f = obs["v_fix"][0][-1].permute(1, 2, 0).numpy()
        image_g = obs["v_gripper"][0][-1].permute(1, 2, 0).numpy()
        if save_video:
            video_writer.write(cv2.cvtColor((255 * image_f).astype(np.uint8), cv2.COLOR_RGB2BGR))
        fix_img_list.append((255 * image_f).astype(np.int_))
        image = np.concatenate([image_f, image_g], axis=0)
        plt.imshow(image)
        plt.draw()
        plt.pause(0.2)
        plt.clf()
    if save_video:
        video_writer.release()
    plt.close()
    # compare_demo_gen(batch, actions)
    plt.figure(figsize=(16, 16))
    for i in range(7):
        # plot the position and quaternion
        plt.subplot(8, 1, i + 1)
        plt.plot(demo_state_list[:, i], c='r', alpha=0.2, label='demo')
        plt.plot(gen_state_list[:, i], c='b', alpha=0.2, label='gen')
        if i > 2:
            plt.ylim([-1.1, 1.1])
        else:
            plt.ylim([0., 0.6])

        plt.legend()
    plt.subplot(8, 1, 8)
    plt.plot(gen_state_list[:, 7], c='b', alpha=0.2, label='gen')
    plt.plot(demo_grip_list, c='r', alpha=0.2, label='demo')
    plt.show()


if __name__ == '__main__':
    device = torch.device('cuda')

    cfg = OmegaConf.load('diffpol_rotatemug.yaml')
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

    add_info = ''

    checkpoints_dir = cfg.diffusion_model

    action_dim = 3 + 4 + 1
    vision_feature_dim = 512 + 8
    obs_horizon = 2
    n_ref = cfg.n_ref
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
        num_diffusion_iters = 10
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
        num_diffusion_iters = 10
    else:
        assert False, 'Wrong Diffusion Model Type'
    vision_encoder = get_resnet('resnet18')
    vision_encoder = replace_bn_with_gn(vision_encoder)
    nets = torch.nn.ModuleDict({'vision_encoder': vision_encoder, 'noise_pred_net': noise_pred_net})
    nets.to(device)
    nets.load_state_dict(torch.load('DDIM_hard/DDIM_harddata_.pt'))

    data_utils = RotateMugDataUtils(cfg=cfg, nets=nets)

    infer_whole_traj(data_loadar=val_loader, nets=nets, data_utils=data_utils, noise_scheduler=noise_scheduler,
                     action_horizon=8, save_video=False, no_crop=False, num_diffusion_iters=num_diffusion_iters)

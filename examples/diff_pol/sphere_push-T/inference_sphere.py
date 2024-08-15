import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from manifm.vision.resnet_models import get_resnet, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.dataset.pusht_state_dataset import normalize_data, unnormalize_data
from omegaconf import OmegaConf
from manifm.datasets import _get_dataset
import collections
from tqdm import tqdm
from IPython.display import Video
from skvideo.io import vwrite
import matplotlib

matplotlib.use('TkAgg')
import os
import sys

sys.path.append('/home/dia1rng/safe_flow_motion_policy/flow-matching-policies/examples/datasets')
from create_pusht_sphere_image_dataset import project_sphere_action_back, \
    project_to_sphere_agent_pos, plt3d_sphere_project_img
import torchvision.transforms as Transform
import cv2
import shutil
from manifm.manifolds import Sphere, ProductManifoldTrajectories


def infer_sphere(nets, noise_scheduler, seed=66, action_horizon=8, crop=True, scale=1.5,
                 num_diffusion_iters=100, save_folder='', device=torch.device('cuda'), seq_unpack=True, sphere_prior=True):
    if not os.path.exists(save_folder + '/seed' + str(seed) + '/sp_img'):
        os.mkdir(save_folder + '/seed' + str(seed))
        os.mkdir(save_folder + '/seed' + str(seed) + '/sp_img')
        os.mkdir(save_folder + '/seed' + str(seed) + '/sp_infer')
    manifold = [(Sphere(), 3) for _ in range(cfg.n_pred)]
    manifold = ProductManifoldTrajectories(*manifold)
    max_steps = 20
    env = PushTImageEnv()
    env.seed(seed)
    obs = env.reset()
    obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
    imgs = [env.render(mode='rgb_array')]
    rewards = list()
    done = False
    step_idx = 0
    with tqdm(total=max_steps, desc="Eval PushTImageEnv") as pbar:
        while not done:
            B = 1
            # stack the last obs_horizon number of observations
            images = np.stack([x['image'] for x in obs_deque])
            agent_poses = np.stack([x['agent_pos'] for x in obs_deque])

            # normalize & project to sphere
            agent_poses = (agent_poses - 256) / 256
            sp_agent_poses = project_to_sphere_agent_pos(agent_poses * scale, r=1)

            # project image to sphere
            nimages = images
            sp_images = np.zeros((2, 3, 100, 100))
            for id in range(nimages.shape[0]):
                cur_img = nimages[id]
                sp_cur_img = plt3d_sphere_project_img(255 * np.moveaxis(cur_img, 0, -1), scale=scale,
                                                      save_dir=save_folder + '/seed' + str(
                                                          seed) + '/sp_infer' + '/infer' + str(id))
                # sp_cur_img = sphere_project_img(255 * np.moveaxis(cur_img, 0, -1))
                sp_images[id] = np.moveaxis(sp_cur_img, -1, 0) / 255

            sp_images = torch.from_numpy(sp_images).to(device, dtype=torch.float32).unsqueeze(0)
            if crop:
                crop_transform = Transform.CenterCrop((84, 84))
                sp_images = crop_transform(sp_images)
            sp_agent_poses = torch.from_numpy(sp_agent_poses).to(device, dtype=torch.float32)
            # (2,2)

            # infer action
            with torch.no_grad():
                # get image features
                image_features = nets['vision_encoder'](sp_images.flatten(end_dim=1))
                # image_features = image_features.reshape(*sp_images.shape[:2], -1).flatten(start_dim=1)
                # (2,512)

                if seq_unpack:
                    # concat with low-dim observations
                    obs_features = torch.cat([image_features, sp_agent_poses], dim=-1).unsqueeze(0)
                else:
                    sp_agent_poses = sp_agent_poses.reshape((B, cfg.n_ref * 3))
                    image_features = image_features.reshape((B, cfg.n_ref * 512))
                    obs_features = torch.cat([image_features, sp_agent_poses], dim=-1)
                # reshape observation to (B,obs_horizon*obs_dim)
                obs_cond = obs_features.flatten(start_dim=1)

                if not sphere_prior:
                    # initialize action from Guassian noise
                    noisy_action = torch.randn(
                        (B, 16, action_dim), device=device)
                    naction = noisy_action
                else:
                    noisy_action = manifold.random_base(B, cfg.n_pred * 3).to(device).reshape((B, cfg.n_pred, 3))
                    naction = noisy_action
                # init scheduler
                noise_scheduler.set_timesteps(num_diffusion_iters)

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

            # unnormalize action
            naction = naction.detach().to('cpu').numpy()
            # (B, pred_horizon, action_dim)
            naction = naction[0]
            naction = naction / np.expand_dims(np.linalg.norm(naction, axis=1), axis=-1)
            action_pred = project_sphere_action_back(naction, r=1)
            action_pred = (action_pred / scale + 1) * 256

            # only take action_horizon number of actions
            start = obs_horizon - 1
            end = start + action_horizon
            action = action_pred[start:end, :]
            # (action_horizon, action_dim)

            # execute action_horizon number of steps
            # without replanning
            for i in range(len(action)):
                # stepping env
                obs, reward, done, info = env.step(action[i])
                # save observations
                obs_deque.append(obs)
                # and reward/vis
                rewards.append(reward)
                cur_record_img = env.render(mode='rgb_array')
                imgs.append(cur_record_img)
                plt3d_sphere_project_img(cur_record_img, scale=scale,
                                         save_dir=save_folder + '/seed' + str(seed) + '/sp_img' + '/' + str(
                                             step_idx))

                # update progress bar
                step_idx += 1
                pbar.update(1)
                pbar.set_postfix(reward=reward)
                if step_idx > max_steps:
                    done = True
                if done:
                    break

    # print out the maximum target coverage
    print('Score: ', max(rewards))

    if not os.path.exists(save_folder + '/seed' + str(seed)):
        os.mkdir(save_folder + '/seed' + str(seed))

    # visualize
    vwrite(
        save_folder + '/seed' + str(seed) + '/sphere_' + str(action_horizon) + 'seed' + str(
            seed) + '.mp4', imgs)
    Video(save_folder + '/seed' + str(seed) + '/sphere_' + str(action_horizon) + str(seed) + '.mp4',
          embed=True, width=256, height=256)

    make_video(save_folder + '/seed' + str(seed) + '/sp_img')
    shutil.rmtree(save_folder + '/seed' + str(seed) + '/sp_img')
    return max(rewards)


def make_video(datadir):
    file_list = os.listdir(datadir)
    fps = 10
    video_name = datadir + '.mp4'
    file_list.sort()
    file_list = get_ordered_file_list(file_list)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_name, fourcc, fps, (100, 100))
    for file in file_list:
        video_writer.write(cv2.imread(datadir + '/' + file))
        # fig_list.append(cv2.imread(datadir + '/' + file))
    video_writer.release()
    # vwrite(datadir + '/sp_pusht.mp4', fig_list)
    # Video(datadir + '/sp_pusht.mp4', embed=True, width=256,
    #       height=256)


def get_ordered_file_list(file_list):
    num_list = [int(file[:-4]) for file in file_list]
    sorted_lists = sorted(zip(num_list, file_list))
    _, file_list = zip(*sorted_lists)
    return file_list


if __name__ == '__main__':
    cfg = OmegaConf.load('diffpol_sphere_vision_pusht.yaml')
    cfg.dim = 3
    device = torch.device('cuda')
    dataset, _ = _get_dataset(cfg)
    stats = dataset.stats
    action_dim = 3
    obs_dim = 512 + action_dim
    obs_horizon = 2
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * obs_horizon,
        down_dims=[256, 512, 1024],
    )
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
    vision_encoder = get_resnet('resnet18')
    vision_encoder = replace_bn_with_gn(vision_encoder)
    nets = torch.nn.ModuleDict({'vision_encoder': vision_encoder, 'noise_pred_net': noise_pred_net})
    nets.load_state_dict(torch.load('500epochtrain/DDIM/DDIM.pt'))
    nets.to(device)
    reward_list = []
    save_folder = '500epochtrain/DDIM'
    for seed in range(1, 51):
        cur_reward = infer_sphere(nets=nets, noise_scheduler=noise_scheduler, action_horizon=8, num_diffusion_iters=5,
                     save_folder=save_folder, scale=1.5, seed=66 * seed, seq_unpack=False)
        reward_list.append(cur_reward)
        np.save(save_folder + '/reward_list' + '' + '.npy', reward_list)
        print('*********************reward mean ' + str(np.array(reward_list).mean()) + '************************')
    reward_list = np.array(reward_list)

    print(reward_list)
    print(reward_list.mean())

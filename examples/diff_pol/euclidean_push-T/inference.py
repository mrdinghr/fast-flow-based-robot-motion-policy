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
import torchvision.transforms as Transform


def infer(nets, stats, noise_scheduler, seed=66, save_video=False, action_horizon=8, crop=True):
    max_steps = 500
    env = PushTImageEnv()
    env.seed(seed)
    stats['agent_pos'] = {'min': stats['state']['min'][:2], 'max': stats['state']['max'][:2]}
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

            # normalize observation
            nagent_poses = normalize_data(agent_poses, stats=stats['agent_pos'])
            # images are already normalized to [0,1]
            nimages = images

            # device transfer
            nimages = torch.from_numpy(nimages).to(device, dtype=torch.float32)
            # (2,3,96,96)
            if crop:
                crop_transform = Transform.CenterCrop((84, 84))
                nimages = crop_transform(nimages)
            nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)
            # (2,2)

            # infer action
            with torch.no_grad():
                # get image features
                image_features = nets['vision_encoder'](nimages)
                # (2,512)

                # concat with low-dim observations
                obs_features = torch.cat([image_features, nagent_poses], dim=-1)

                # reshape observation to (B,obs_horizon*obs_dim)
                obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

                # initialize action from Guassian noise
                noisy_action = torch.randn(
                    (B, 16, action_dim), device=device)
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
            action_pred = unnormalize_data(naction, stats=stats['action'])

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
                imgs.append(env.render(mode='rgb_array'))

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
    if save_video:
        # visualize
        vwrite('act' + str(action_horizon) + 'seed' + str(seed) + '.mp4', imgs)
        Video('act' + str(action_horizon) + str(seed) + '.mp4', embed=True, width=256, height=256)
    return max(rewards)


if __name__ == '__main__':
    cfg = OmegaConf.load('diffpol_euclidean_vision_pusht.yaml')
    num_diffusion_iters = 10
    device = torch.device('cuda')
    dataset, _ = _get_dataset(cfg)
    stats = dataset.stats
    action_dim = 2
    obs_dim = 512 + 2
    obs_horizon = 2
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * obs_horizon,
        down_dims=[256, 512, 1024],
    )
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
    vision_encoder = get_resnet('resnet18')
    vision_encoder = replace_bn_with_gn(vision_encoder)
    nets = torch.nn.ModuleDict({'vision_encoder': vision_encoder, 'noise_pred_net': noise_pred_net})
    nets.load_state_dict(torch.load('Model/DDPM.pt'))
    nets.to(device)
    # infer(nets=nets, stats=stats, noise_scheduler=noise_scheduler)

    reward_list = []
    for seed in range(1, 51):
        reward = infer(nets=nets, stats=stats, noise_scheduler=noise_scheduler, seed=seed * 66, save_video=True)
        reward_list.append(reward)
        np.save('reward_list' + '' + '.npy', reward_list)
        print('*********************reward mean ' + str(np.array(reward_list).mean()) + '************************')
    reward_list = np.array(reward_list)
    # np.save(checkpoints_dir + '/ode100_t1.5_seed1-50' + '/reward_list' + '' + '.npy', reward_list)
    print(reward_list)
    print(reward_list.mean())

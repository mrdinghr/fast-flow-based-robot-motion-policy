import os.path
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from tqdm.auto import tqdm
from IPython.display import Video
from skvideo.io import vwrite
import collections
from glob import glob

import matplotlib

matplotlib.use('Agg')

from manifm.datasets import _get_dataset
import sys

sys.path.append('/home/dia1rng/hackathon/flow-matching-policies/stable_flow')
from stable_model_vision_trajs_pl_learntau import SRFMVisionResnetTrajsModuleLearnTau
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.dataset.pusht_state_dataset import normalize_data, unnormalize_data

import torchvision.transforms as Transform

# if torch.cuda.is_available():
#     device = torch.cuda.current_device()
# else:
#     device = 'cpu'
device = torch.device('cuda')


def inference(stats, model, obs_horizon, pred_horizon, action_horizon, seed=100000, crop=False, adp=False, ode_steps=10, new_unpack=True):
    if not os.path.exists(checkpoints_dir + '/automatic_dt/ode_step' + str(ode_steps)):
        os.mkdir(checkpoints_dir + '/automatic_dt/ode_step' + str(ode_steps))

    # limit enviornment interaction to 200 steps before termination
    max_steps = 500
    env = PushTImageEnv()
    # use a seed >200 to avoid initial states seen in the training dataset
    env.seed(seed)
    stats['agent_pos'] = {'min': stats['state']['min'][:2], 'max': stats['state']['max'][:2]}
    # stats['images'] = {'min': stats['img']['min'].repeat(96 * 96).reshape((3, 96, 96)), 'max': stats['img']['max'].repeat(96 *96).reshape((3, 96, 96))}

    # get first observation
    # obs, info = env.reset()
    obs = env.reset()

    # keep a queue of last 2 steps of observations
    obs_deque = collections.deque(
        [obs] * obs_horizon, maxlen=obs_horizon)
    # save visualization and rewards
    imgs = [env.render(mode='rgb_array')]
    rewards = list()
    done = False
    step_idx = 0

    with tqdm(total=max_steps, desc="Eval PushTStateEnv") as pbar:
        while not done:
            B = 1
            # stack the last obs_horizon number of observations
            images = np.stack([x['image'] for x in obs_deque])
            agent_poses = np.stack([x['agent_pos'] for x in obs_deque])
            # # normalize observation
            nagent_poses = normalize_data(agent_poses, stats=stats['agent_pos'])
            # # images are already normalized to [0,1]
            # nimages = normalize_data(images * 255, stats=stats['images'])
            nimages = images
            # nagent_poses = agent_poses
            # nimages = images * 255

            # device transfer
            nimages = torch.from_numpy(nimages).to(device, dtype=torch.float32).unsqueeze(0).to(model.device)
            # (2,3,96,96)
            if crop:
                crop_transform = Transform.CenterCrop((84, 84))
                nimages = crop_transform(nimages)
            nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32).to(model.device)
            if not new_unpack:
                nagent_poses = nagent_poses.reshape((B, model.n_ref * model.dim))
            # (2,2)

            with torch.no_grad():
                # get image features
                if new_unpack:
                    image_features = model.new_image_to_features(nimages).squeeze(0)
                else:
                    image_features = model.image_to_features(nimages)

                # concat with low-dim observations
                obs_features = torch.cat([image_features, nagent_poses], dim=-1)

                # reshape observation to (B,obs_horizon*obs_dim)
                obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1).to(model.device)

                # Flow matching action
                start_time = time.time()
                ode_traj = model.sample_all(B, obs_cond.device, obs_cond,
                                            torch.zeros(B, 0, dtype=obs_cond.dtype, device=obs_cond.device),
                                            different_sample=True, adp=adp, ode_steps=ode_steps)
                print('inference uses ' + str(time.time() - start_time))
                print('solved tau ' + str(ode_traj[:, 0, -1]))
                naction = ode_traj[-1][None][..., :model.dim * pred_horizon].squeeze().reshape(B, pred_horizon,
                                                                                               model.dim)

            # unnormalize action
            naction = naction.detach().to('cpu').numpy()
            # (B, pred_horizon, action_dim)
            naction = naction[0]
            action_pred = unnormalize_data(naction, stats=stats['action'])
            # action_pred = naction

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
    vwrite(
        checkpoints_dir + '/automatic_dt/ode_step' + str(ode_steps) + '/seed' + str(seed) + '.mp4',
        imgs)
    Video(checkpoints_dir + '/automatic_dt/ode_step' + str(ode_steps) + '/seed' + str(seed) + '.mp4',
          embed=True, width=256, height=256)

    return max(rewards)


def inference_band(stats, model, obs_horizon, pred_horizon, action_horizon, seed=100000, crop=False, adp=False, ode_steps=10, new_unpack=False):
    if not os.path.exists(checkpoints_dir + '/automatic_dt/ode_step' + str(ode_steps)):
        os.mkdir(checkpoints_dir + '/automatic_dt/ode_step' + str(ode_steps))

    # limit enviornment interaction to 200 steps before termination
    max_steps = 500
    env = PushTImageEnv()
    # use a seed >200 to avoid initial states seen in the training dataset
    env.seed(seed)
    stats['agent_pos'] = {'min': stats['state']['min'][:2], 'max': stats['state']['max'][:2]}
    # stats['images'] = {'min': stats['img']['min'].repeat(96 * 96).reshape((3, 96, 96)), 'max': stats['img']['max'].repeat(96 *96).reshape((3, 96, 96))}

    # get first observation
    # obs, info = env.reset()
    obs = env.reset()

    # keep a queue of last 2 steps of observations
    obs_deque = collections.deque(
        [obs] * (model.w_cond + 1), maxlen=model.w_cond + 1)
    # save visualization and rewards
    imgs = [env.render(mode='rgb_array')]
    rewards = list()
    done = False
    step_idx = 0

    with tqdm(total=max_steps, desc="Eval PushTStateEnv") as pbar:
        while not done:
            B = 1
            cond_id = np.random.randint(model.w_cond)
            # stack the last obs_horizon number of observations
            cur_img = torch.from_numpy(obs_deque[-1]['image']).to(device, dtype=torch.float32).unsqueeze(0)
            sample_img = torch.from_numpy(obs_deque[cond_id]['image']).to(device, dtype=torch.float32).unsqueeze(0)
            ref_img = torch.cat([sample_img, cur_img]).unsqueeze(0)

            cur_pose = obs_deque[-1]['agent_pos']
            sample_pose = obs_deque[cond_id]['agent_pos']
            agent_poses = np.row_stack([sample_pose, cur_pose])
            nagent_poses = normalize_data(agent_poses, stats=stats['agent_pos'])
            if crop:
                crop_transform = Transform.CenterCrop((84, 84))
                nimages = crop_transform(ref_img)
            nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32).to(model.device)
            if not new_unpack:
                nagent_poses = nagent_poses.reshape((B, model.n_ref * model.dim))
            # (2,2)

            with torch.no_grad():
                # get image features
                if new_unpack:
                    image_features = model.new_image_to_features(nimages).squeeze(0)
                else:
                    image_features = model.image_to_features(nimages)

                # concat with low-dim observations
                obs_features = torch.cat([image_features, nagent_poses,
                                          torch.tensor([model.w_cond - cond_id]).to(model.device).unsqueeze(0)],
                                         dim=-1).float()
                # reshape observation to (B,obs_horizon*obs_dim)
                obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

                # Flow matching action
                start_time = time.time()
                ode_traj = model.sample_all(B, obs_cond.device, obs_cond,
                                            torch.zeros(B, 0, dtype=obs_cond.dtype, device=obs_cond.device),
                                            different_sample=True, adp=adp, ode_steps=ode_steps)
                print('inference uses ' + str(time.time() - start_time))
                print('solved tau ' + str(ode_traj[:, 0, -1]))
                naction = ode_traj[-1][None][..., :model.dim * pred_horizon].squeeze().reshape(B, pred_horizon,
                                                                                               model.dim)

            # unnormalize action
            naction = naction.detach().to('cpu').numpy()
            # (B, pred_horizon, action_dim)
            naction = naction[0]
            action_pred = unnormalize_data(naction, stats=stats['action'])
            # action_pred = naction

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
    # vwrite(
    #     checkpoints_dir + '/automatic_dt/ode_step' + str(ode_steps) + '/seed' + str(seed) + '.mp4',
    #     imgs)
    # Video(checkpoints_dir + '/automatic_dt/ode_step' + str(ode_steps) + '/seed' + str(seed) + '.mp4',
    #       embed=True, width=256, height=256)

    return max(rewards)


if __name__ == '__main__':
    # torch.manual_seed(3047)
    # Load config
    add_info = '_lambdataux25_obs8'  # _learntau_new_encoder_lambdatau
    cfg = OmegaConf.load('refcond_srfm_euclidean_learntau_vision_pusht.yaml')
    # checkpoints_dir = "checkpoints/checkpoints_rfm_" + cfg.data + "_" + cfg.letter[0] + "_n" + str(cfg.n_pred)
    checkpoints_dir = "checkpoints/checkpoints_rfm_" + cfg.data + \
                      "_n" + str(cfg.n_pred) + "_r" + str(cfg.n_ref) + "_c" + str(cfg.n_cond) + "_w" + str(
        cfg.w_cond) + cfg.model_type + add_info

    # Number of steps and actions per step
    # n_actions = 1
    # n_steps = int(200 / n_actions)

    # Load dataset
    dataset, _ = _get_dataset(cfg)

    # Construct model
    model = SRFMVisionResnetTrajsModuleLearnTau(cfg)
    print(model)
    model.small_var = True
    best_checkpoint = glob(checkpoints_dir + "/**/epoch**.ckpt", recursive=True)[0]
    last_checkpoint = './' + checkpoints_dir + '/last.ckpt'
    # model_checkpoint = './' + checkpoints_dir + '/model.ckpt'
    model = model.load_from_checkpoint(best_checkpoint, cfg=cfg)
    model.to(torch.device('cuda'))  # cuda  cpu

    # Training data statistics (min, max) for each dim
    stats = dataset.stats

    # inference(stats, model, obs_horizon=cfg.n_ref, pred_horizon=cfg.n_pred, action_horizon=int(8),
    #           seed=66)
    if not os.path.exists(checkpoints_dir + '/automatic_dt'):
        os.mkdir(checkpoints_dir + '/automatic_dt')
    for ode_steps in [1, 3, 5, 10]:
        reward_list = []
        # ode_steps = 1
        for seed in tqdm(range(1, 51)):
            if 'band' in cfg.data:
                reward = inference_band(stats, model, obs_horizon=cfg.n_ref, pred_horizon=cfg.n_pred,
                                        action_horizon=int(8),
                                        seed=seed * 66, crop=True, adp=False, ode_steps=ode_steps, new_unpack=False)
            else:
                reward = inference(stats, model, obs_horizon=cfg.n_ref, pred_horizon=cfg.n_pred, action_horizon=int(8),
                                   seed=seed * 66, crop=True, adp=False, ode_steps=ode_steps, new_unpack=False)
            reward_list.append(reward)
            np.save(checkpoints_dir + '/automatic_dt/ode_step' + str(ode_steps) + '/reward_list' + '' + '.npy', reward_list)
            print('*********************reward mean ' + str(np.array(reward_list).mean()) + '************************')
        reward_list = np.array(reward_list)
        # np.save(checkpoints_dir + '/ode100_t1.5_seed1-50' + '/reward_list' + '' + '.npy', reward_list)
        print(reward_list)
        print(reward_list.mean())

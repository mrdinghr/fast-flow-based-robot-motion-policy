import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

import collections
from glob import glob

import matplotlib

matplotlib.use('TkAgg')

from manifm.datasets import _get_dataset
from manifm.model_trajectories_vision_resnet_pl import ManifoldVisionTrajectoriesResNetFMLitModule
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
import torchvision.transforms as Transform
from diffusion_policy.dataset.pusht_state_dataset import normalize_data, unnormalize_data
import time
from tqdm.auto import tqdm


def inference(stats, model, obs_horizon, pred_horizon, action_horizon, seed=100000, crop=False, ode_steps=10,
              new_unpack=False, ot_prior=False, visualization_onscreen=False):
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

    rewards = list()
    done = False
    step_idx = 0
    img_id = 0
    img = env.render(mode='rgb_array')

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
            nimages = torch.from_numpy(nimages).to(device, dtype=torch.float32).unsqueeze(0)
            # (2,3,96,96)
            if crop:
                crop_transform = Transform.CenterCrop((84, 84))
                nimages = crop_transform(nimages).float()
            nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)
            if not new_unpack:
                nagent_poses = nagent_poses.reshape((B, model.n_ref * model.dim))
            # (2,2)

            with torch.no_grad():
                # get image features
                if not new_unpack:
                    image_features = model.image_to_features(nimages)
                else:
                    image_features = model.new_image_to_features(nimages).squeeze(0)

                # concat with low-dim observations
                obs_features = torch.cat([image_features, nagent_poses], dim=-1)

                # reshape observation to (B,obs_horizon*obs_dim)
                obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

                # Flow matching action
                start_time = time.time()
                if ot_prior:
                    ot_x0 = nagent_poses.repeat(1, int(model.n_pred / model.n_ref))
                    ot_x0 += torch.normal(mean=0, std=std, size=ot_x0.shape).to(model.device)
                    ode_traj = model.sample_all(B, obs_cond.device, obs_cond,
                                                torch.zeros(B, 0, dtype=obs_cond.dtype, device=obs_cond.device),
                                                different_sample=True, ode_steps=ode_steps + 1, x0=ot_x0)
                else:
                    ode_traj = model.sample_all(B, obs_cond.device, obs_cond,
                                                torch.zeros(B, 0, dtype=obs_cond.dtype, device=obs_cond.device),
                                                different_sample=True, ode_steps=ode_steps + 1)
                print('inference uses ' + str(time.time() - start_time))
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
            img_id = plot_vector_field(model=model, stats=stats, sample_num_per_axis=11, env=env, x_ref=obs_cond,
                                       img_id=img_id, visualization_onscreen=visualization_onscreen)

            # execute action_horizon number of steps
            # without replanning
            for i in range(len(action)):
                # stepping env
                obs, reward, done, info = env.step(action[i])
                obs_deque.append(obs)
                # and reward/vis
                rewards.append(reward)
                img = env.render(mode='rgb_array')
                plt.imshow(img)
                plt.axis('off')
                plt.savefig(save_folder + '/' + str(img_id) + '.png')
                img_id += 1
                if visualization_onscreen:
                    plt.draw()
                    plt.pause(0.1)
                    plt.clf()
                else:
                    plt.close()
                # plt.close()
                # imgs.append(env.render(mode='rgb_array'))

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
    return max(rewards)


def plot_vector_field(model, stats=None, sample_num_per_axis=11, env=None, x_ref=None, img_id=0,
                      visualization_onscreen=False):
    x0 = get_x0(sample_num_per_axis=sample_num_per_axis, stats=stats)
    t = torch.zeros((sample_num_per_axis ** 2, 1)).to(device)
    model.model.vecfield.vecfield.unet.global_cond = x_ref.repeat(len(x0), 1)
    vector_field = model.vecfield(t, x0)
    for dim in range(8):
        cur_dim_vector_field = vector_field[:, dim * 2: dim * 2 + 2].detach().cpu().numpy()
        img_id = visualize_vector_field(cur_dim_vector_field, env_img=env.render(mode='rgb_array'), img_id=img_id,
                                        visualization_onscreen=visualization_onscreen)
    return img_id


def get_x0(xy_lim=[0, 96], sample_num_per_axis=11, x0_dim=32, stats=None):
    xs = torch.linspace(xy_lim[0], xy_lim[1], steps=sample_num_per_axis)
    ys = torch.linspace(xy_lim[0], xy_lim[1], steps=sample_num_per_axis)
    xm, ym = torch.meshgrid(xs, ys, indexing='xy')
    x = xm.reshape((sample_num_per_axis ** 2, 1))
    y = ym.reshape(sample_num_per_axis ** 2, 1)
    x0 = torch.hstack([x, y]).to(device)
    x0 = x0 * 512 / 96
    x0 = normalize_data(x0.cpu().numpy(), stats=stats['agent_pos'])
    x0 = torch.from_numpy(x0).to(device)
    # x0_zero_rest_dim = torch.zeros(int(sample_num_per_axis * sample_num_per_axis), x0_dim - 2).to(device)
    return x0.repeat(1, 16)


def visualize_vector_field(vector_field, xy_lim=[0, 96], sample_num_per_axis=11, env_img=None, img_id=0,
                           visualization_onscreen=False):
    xs = torch.linspace(xy_lim[0], xy_lim[1], steps=sample_num_per_axis)
    ys = torch.linspace(xy_lim[0], xy_lim[1], steps=sample_num_per_axis)
    xm, ym = torch.meshgrid(xs, ys, indexing='xy')
    x = xm.reshape((sample_num_per_axis ** 2, 1))
    y = ym.reshape(sample_num_per_axis ** 2, 1)
    xy = torch.cat((x, y), dim=1)
    plt.imshow(env_img)
    plt.quiver(xy[:, 0], xy[:, 1], vector_field[:, 0], -vector_field[:, 1], cmap='viridis')
    # plt.title('vanilla RFMP')
    plt.axis('off')
    plt.savefig(save_folder + '/' + str(img_id) + '.png')
    if visualization_onscreen:
        plt.draw()
        plt.pause(0.1)
        plt.clf()
    else:
        plt.close()
    # plt.close()
    # plt.show()
    return img_id + 1


if __name__ == '__main__':
    device = torch.device('cuda')
    std = 0.1
    add_info = '_otflow_refprior_std' + str(std)
    # add_info = '_resnet_imgcrop'
    cfg = OmegaConf.load('refcond_rfm_euclidean_vision_pusht.yaml')
    dataset, _ = _get_dataset(cfg)
    stats = dataset.stats
    stats['agent_pos'] = {'min': stats['state']['min'][:2], 'max': stats['state']['max'][:2]}
    checkpoints_dir = "checkpoints/checkpoints_rfm_" + cfg.data + \
                      "_n" + str(cfg.n_pred) + "_r" + str(cfg.n_ref) + "_c" + str(cfg.n_cond) + "_w" + str(
        cfg.w_cond) + cfg.model_type + add_info
    save_folder = checkpoints_dir + '/vector_field_test'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    model = ManifoldVisionTrajectoriesResNetFMLitModule(cfg)
    model.small_var = False
    best_checkpoint = glob(checkpoints_dir + "/**/epoch**.ckpt", recursive=True)[0]
    last_checkpoint = './' + checkpoints_dir + '/last.ckpt'
    model_checkpoint = './' + checkpoints_dir + '/model.ckpt'
    model = model.load_from_checkpoint(best_checkpoint, cfg=cfg)
    model.to(torch.device('cuda'))
    inference(stats, model, obs_horizon=cfg.n_ref, pred_horizon=cfg.n_pred,
              action_horizon=int(8),
              seed=66 * 1, crop=True, ode_steps=5 + 1, ot_prior=True, visualization_onscreen=False)

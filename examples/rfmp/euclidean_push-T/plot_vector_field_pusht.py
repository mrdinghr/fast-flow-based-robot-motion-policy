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


def plot_vector_field(model, seed=66, stats=None, sample_num_per_axis=11):
    env = PushTImageEnv()
    env.seed(seed)
    obs = env.reset()
    obs_deque = collections.deque(
        [obs] * model.n_ref, maxlen=model.n_ref)
    x_ref = get_xref(obs_deque, stats, model)
    x_ref = x_ref.repeat(sample_num_per_axis ** 2, 1)
    x0 = get_x0(sample_num_per_axis=sample_num_per_axis, stats=stats)
    t = torch.zeros((sample_num_per_axis ** 2, 1)).to(device)
    model.model.vecfield.vecfield.unet.global_cond = x_ref
    vector_field = model.vecfield(t, x0)
    vector_field_first_dim = vector_field[:, :2].detach().cpu().numpy()
    visualize_vector_field(vector_field_first_dim, env_img=env.render(mode='rgb_array'))


@torch.no_grad()
def get_xref(obs_deque, stats, model):
    images = np.stack([x['image'] for x in obs_deque])

    nimages = torch.from_numpy(images).to(device, dtype=torch.float32).unsqueeze(0).to(model.device)
    crop_transform = Transform.CenterCrop((84, 84))
    nimages = crop_transform(nimages)
    agent_poses = np.stack([x['agent_pos'] for x in obs_deque])
    nagent_poses = normalize_data(agent_poses, stats=stats['agent_pos'])
    nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32).to(model.device)
    nagent_poses = nagent_poses.reshape((1, model.n_ref * model.dim))
    image_features = model.image_to_features(nimages)
    obs_features = torch.cat([image_features, nagent_poses], dim=-1)
    obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1).to(model.device)
    return obs_cond


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
    x0_zero_rest_dim = torch.zeros(int(sample_num_per_axis * sample_num_per_axis), x0_dim - 2).to(device)
    return torch.cat((x0, x0_zero_rest_dim), dim=1)


def visualize_vector_field(vector_field, xy_lim=[0, 96], sample_num_per_axis=11, env_img=None):
    xs = torch.linspace(xy_lim[0], xy_lim[1], steps=sample_num_per_axis)
    ys = torch.linspace(xy_lim[0], xy_lim[1], steps=sample_num_per_axis)
    xm, ym = torch.meshgrid(xs, ys, indexing='xy')
    x = xm.reshape((sample_num_per_axis ** 2, 1))
    y = ym.reshape(sample_num_per_axis ** 2, 1)
    xy = torch.cat((x, y), dim=1)
    plt.imshow(env_img)
    plt.quiver(xy[:, 0], xy[:, 1], vector_field[:, 0], -vector_field[:, 1], cmap='viridis')
    plt.title('vanilla RFMP')
    plt.axis('off')
    plt.savefig(checkpoints_dir + '/' + add_info + 'vector_field.png',)
    plt.show()


if __name__ == '__main__':
    device = torch.device('cuda')
    # add_info = '_otflow_refprior_std2.0'
    add_info = '_resnet_imgcrop'
    cfg = OmegaConf.load('refcond_rfm_euclidean_vision_pusht.yaml')
    dataset, _ = _get_dataset(cfg)
    stats = dataset.stats
    stats['agent_pos'] = {'min': stats['state']['min'][:2], 'max': stats['state']['max'][:2]}
    checkpoints_dir = "checkpoints/checkpoints_rfm_" + cfg.data + \
                      "_n" + str(cfg.n_pred) + "_r" + str(cfg.n_ref) + "_c" + str(cfg.n_cond) + "_w" + str(
        cfg.w_cond) + cfg.model_type + add_info
    model = ManifoldVisionTrajectoriesResNetFMLitModule(cfg)
    best_checkpoint = glob(checkpoints_dir + "/**/epoch**.ckpt", recursive=True)[0]
    last_checkpoint = './' + checkpoints_dir + '/last.ckpt'
    model_checkpoint = './' + checkpoints_dir + '/model.ckpt'
    model = model.load_from_checkpoint(best_checkpoint, cfg=cfg)
    model.to(torch.device('cuda'))
    plot_vector_field(model, stats=stats, seed=666)

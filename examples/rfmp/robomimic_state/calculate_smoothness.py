import os.path

import numpy as np
from omegaconf import OmegaConf
import torch
from manifm.manifolds import Sphere


device = torch.device('cuda')


def cal_transition_smooth(trajectory):
    trajectory = torch.from_numpy(trajectory).to(device)
    vel = torch.diff(trajectory, axis=0)
    acc = torch.diff(vel, axis=0)
    jerk = torch.diff(acc, axis=0)
    jerk_euc = torch.sum(torch.abs(torch.sum(jerk, axis=0))).cpu().numpy()
    return jerk_euc


def cal_rotation_smooth(rotate_traj):
    manifold = Sphere()
    rotate_traj = torch.from_numpy(rotate_traj)
    vel = manifold.logmap(rotate_traj[:-1], rotate_traj[1:])
    acc = torch.diff(vel, axis=0)
    jerk_riemannian = torch.diff(acc, axis=0)
    jerk_riemannian = manifold.inner(rotate_traj[:-3], jerk_riemannian, jerk_riemannian).cpu().numpy().sum()
    return jerk_riemannian


def loop_folder_total_smooth(folder, save_folder=None):
    if save_folder is None:
        save_folder = folder
    total_trans_smooth = []
    total_rotate_smooth = []
    for seed in range(1, 51):
        transition_traj_cur_seed = np.load(folder + '/seed' + str(seed * 6) + '_trajtrans.npy')
        rotation_traj_cur_seed = np.load(folder + '/seed' + str(seed * 6) + '_trajrotate.npy')
        trans_smooth_cur_seed = cal_transition_smooth(transition_traj_cur_seed)
        rotate_smooth_cur_seed = cal_rotation_smooth(rotation_traj_cur_seed)
        total_trans_smooth.append(trans_smooth_cur_seed)
        total_rotate_smooth.append(rotate_smooth_cur_seed)
    np.save(save_folder + '/smooth_trans.npy', total_trans_smooth)
    np.save(save_folder + '/smooth_rotate.npy', total_rotate_smooth)
    if folder != save_folder:
        np.save(folder + '/smooth_trans.npy', total_trans_smooth)
        np.save(folder + '/smooth_rotate.npy', total_rotate_smooth)


if __name__ == '__main__':
    cfg = OmegaConf.load('refcond_rfm_robomimic.yaml')
    cfg.task = 'square'
    # can lift square tool_hang  transport
    if cfg.task == 'tool_hang':
        save_gap = 20
    else:
        save_gap = 10
    checkpoints_dir = "checkpoints/checkpoints_rfm_" + cfg.data + \
                      "_n" + str(cfg.n_pred) + "_r" + str(cfg.n_ref) + "_c" + str(cfg.n_cond) + "_w" + str(
        cfg.w_cond) + cfg.model_type + '_task_' + cfg.task + '_saveevery' + str(save_gap) + '_total' + str(save_gap * 5)
    for ode_step in [1, 2, 3, 5, 10]:
        if not os.path.exists('/home/dia1rng/hackathon/flow-matching-policies/plot_utils/robomimic/' + cfg.task + '/rfmp/epoch49'):
            os.mkdir('/home/dia1rng/hackathon/flow-matching-policies/plot_utils/robomimic/' + cfg.task + '/rfmp/epoch49')
        save_folder = '/home/dia1rng/hackathon/flow-matching-policies/plot_utils/robomimic/' + cfg.task + '/rfmp/epoch49/ode' + str(ode_step)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        total_traj_folder = checkpoints_dir + '/epoch' + str(save_gap * 5 - 1) + '/ode' + str(ode_step)
        loop_folder_total_smooth(total_traj_folder, save_folder=save_folder)
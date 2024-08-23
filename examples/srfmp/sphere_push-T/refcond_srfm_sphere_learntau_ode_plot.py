import sys

sys.path.append('/home/dia1rng/hackathon/flow-matching-policies/stable_flow')
from stable_model_vision_trajs_pl_learntau import SRFMVisionResnetTrajsModuleLearnTau

sys.path.append('/home/dia1rng/safe_flow_motion_policy/flow-matching-policies/examples/datasets')
from create_pusht_sphere_image_dataset import project_sphere_action_back, \
    project_to_sphere_agent_pos, plt3d_sphere_project_img
from omegaconf import OmegaConf
from glob import glob
import torch
import numpy as np
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
import collections
import torchvision.transforms as Transform
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def compare_with_dataloadar(model, seed=66, scale=1.5, ode_steps=5):
    # Euclidean Push-T environment initialization
    env = PushTImageEnv()
    env.seed(seed)
    plot_actions_on_env(env, model, scale=scale, ode_steps=ode_steps)


def plot_actions_on_env(env, model, scale=1.5, ode_steps=5, crop=True, seed=66):
    '''
    env: Euclidean PushT env
    model: SRFMP model
    scale: ratio of euclidean pusht env edge length and sphere to project
    ode_steps: ODE solving steps
    crop: whether to crop image
    seed: env initial seed
    '''
    B = 1
    obs = env.reset()
    obs_deque = collections.deque([obs] * model.n_ref, maxlen=model.n_ref)
    images = np.stack([x['image'] for x in obs_deque])
    agent_poses = np.stack([x['agent_pos'] for x in obs_deque])
    agent_poses = (agent_poses - 256) / 256
    sp_agent_poses = project_to_sphere_agent_pos(agent_poses * scale, r=1)
    nimages = images
    sp_images = np.zeros((2, 3, 100, 100))
    for id in range(nimages.shape[0]):
        cur_img = nimages[id]
        sp_cur_img = plt3d_sphere_project_img(255 * np.moveaxis(cur_img, 0, -1), scale=scale,
                                              save_dir=checkpoints_dir + '/plt_ode' '/seed' + str(
                                                  seed) + '/sp_infer' + '/infer' + str(id))
        # sp_cur_img = sphere_project_img(255 * np.moveaxis(cur_img, 0, -1))
        sp_images[id] = np.moveaxis(sp_cur_img, -1, 0) / 255

    sp_images = torch.from_numpy(sp_images).to(model.device, dtype=torch.float32).unsqueeze(0).to(model.device)
    if crop:
        crop_transform = Transform.CenterCrop((84, 84))
        sp_images = crop_transform(sp_images)
    sp_agent_poses = torch.from_numpy(sp_agent_poses).to(model.device, dtype=torch.float32).to(model.device)
    sp_agent_poses = sp_agent_poses.reshape((B, model.n_ref * model.dim))
    with torch.no_grad():
        # get image features
        image_features = model.image_to_features(sp_images)

        # concat with low-dim observations
        obs_features = torch.cat([image_features, sp_agent_poses], dim=-1)

        # reshape observation to (B,obs_horizon*obs_dim)
        obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1).to(model.device)

        # Flow matching action
        start_time = time.time()
        ode_traj = model.sample_all(B, obs_cond.device, obs_cond,
                                    torch.zeros(B, 0, dtype=obs_cond.dtype, device=obs_cond.device),
                                    different_sample=True, adp=False, ode_steps=ode_steps)
        print('infer use ' + str(time.time() - start_time))
    ode_traj = ode_traj.detach().to('cpu').numpy()
    for ode_step, sp_actions in enumerate(ode_traj):
        actions = sp_actions[0, :-1].reshape(model.n_pred, model.dim)
        action_pred = project_sphere_action_back(actions, r=1)
        action_pred = (action_pred / scale + 1) * 256
        env_img = env.render(mode='rgb_array')
        plot_predicted_actions_series_on_iamge(action_pred, env_img, ode_step,
                                               save_name=checkpoints_dir + '/plt_ode' '/seed' + str(seed) + '/' + str(
                                                   ode_step))


def plot_predicted_actions_series_on_iamge(action_series, env_img, ode_step, save_name=''):
    action_series = action_series * 96 / 512
    # fig = plt.figure()
    plt.imshow(env_img)
    cmap = plt.cm.get_cmap('viridis')
    colors = np.linspace(0.1, 1, len(action_series))
    plt.scatter(action_series[..., 0], action_series[..., 1], cmap=cmap, c=colors)
    plt.title('ODE step ' + str(ode_step))
    plt.xlim([-1, 96])
    plt.ylim([96, -1])
    plt.axis('off')
    # plt.savefig(save_name)
    plt.show()


if __name__ == '__main__':
    add_info = '_lambda25_epoch500_new'  # _learntau_new_encoder_lambdatau
    cfg = OmegaConf.load('refcond_srfm_sphere_learntau_vision_pusht.yaml')

    checkpoints_dir = "checkpoints/checkpoints_rfm_" + cfg.data + \
                      "_n" + str(cfg.n_pred) + "_r" + str(cfg.n_ref) + "_c" + str(cfg.n_cond) + "_w" + str(
        cfg.w_cond) + cfg.model_type + add_info
    model = SRFMVisionResnetTrajsModuleLearnTau(cfg)
    best_checkpoint = glob(checkpoints_dir + "/**/epoch**.ckpt", recursive=True)[0]
    last_checkpoint = './' + checkpoints_dir + '/last.ckpt'
    # model_checkpoint = './' + checkpoints_dir + '/model.ckpt'
    model = model.load_from_checkpoint(best_checkpoint, cfg=cfg)
    model.to(torch.device('cuda'))  # cuda  cpu
    compare_with_dataloadar(model, ode_steps=1)

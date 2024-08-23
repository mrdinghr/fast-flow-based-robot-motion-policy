import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from tqdm.auto import tqdm

import collections
from glob import glob

import matplotlib

matplotlib.use('TkAgg')

from manifm.datasets import _get_dataset
import sys

sys.path.append('/home/dia1rng/hackathon/flow-matching-policies/stable_flow')
from stable_model_vision_trajs_pl_learntau import SRFMVisionResnetTrajsModuleLearnTau
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.dataset.pusht_state_dataset import normalize_data, unnormalize_data

import torchvision.transforms as Transform
import cv2

# if torch.cuda.is_available():
#     device = torch.cuda.current_device()
# else:
#     device = 'cpu'
device = 'cpu'


'''
visualize the probability transfer process during test 
SRFMP on Euclidean PushT
'''


def inference(stats, model, obs_horizon, pred_horizon, action_horizon, seed=100000, crop=False, adp=False, ode_steps=10,
              save_folder='', save=False, max_execution_steps=500):
    '''
            stats: statistic data from demonstration, used for normalizetion
            model: SRFMP model
            obs_horizon: observation horizon
            pred_horizon: prediction horizon
            action_horizon: execution horizon
            seed: seed for env initialization
            crop: whether to crop image
            adp: adaptive step size during ODE solving
            ode_steps: ODE solving steps
            save_folder: folder to save experiment data
            max_execution_steps: maximum roll-outs
    '''
    env = PushTImageEnv()
    # use a seed >200 to avoid initial states seen in the training dataset
    env.seed(seed)
    stats['agent_pos'] = {'min': stats['state']['min'][:2], 'max': stats['state']['max'][:2]}

    # get first observation
    obs = env.reset()

    # keep a queue of last 2 steps of observations
    obs_deque = collections.deque(
        [obs] * obs_horizon, maxlen=obs_horizon)
    # save visualization and rewards
    rewards = list()
    done = False
    step_idx = 0

    img_id = 0
    with tqdm(total=max_execution_steps, desc="Eval PushTStateEnv") as pbar:
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
            nagent_poses = nagent_poses.reshape((B, model.n_ref * model.dim))
            # (2,2)

            env_img = env.render(mode='rgb_array')

            with torch.no_grad():
                # get image features
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
                ode_actions = ode_traj[:, :, :-1].reshape(
                    (ode_steps + 1, pred_horizon, model.dim)).detach().cpu().numpy()
                ode_actions = unnormalize_data(ode_actions, stats=stats['action'])
                img_id = plot_ode_solving_process_on_image(ode_actions, env_img, save_folder=save_folder, save=save,
                                                           img_id=img_id)

                naction = ode_traj[-1][None][..., :model.dim * pred_horizon].squeeze().reshape(B, pred_horizon,
                                                                                               model.dim)

            # unnormalize action
            naction = naction.detach().to('cpu').numpy()
            # (B, pred_horizon, action_dim)
            naction = naction[0]
            action_pred = unnormalize_data(naction, stats=stats['action'])
            # action_pred = naction

            plot_predicted_actions_series_on_iamge(action_pred, env_img, save=save, save_folder=save_folder,
                                                   img_id=img_id)
            img_id += 1

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
                env_img = env.render(mode='rgb_array')
                plt.imshow(env_img)
                plt.xlim([-1, 96])
                plt.ylim([96, -1])
                plt.axis('off')
                plt.savefig(save_folder + '/' + str(img_id) + '.png')
                img_id += 1
                plt.close()
                # update progress bar
                step_idx += 1
                pbar.update(1)
                pbar.set_postfix(reward=reward)
                if step_idx > max_execution_steps:
                    done = True
                if done:
                    break

    # print out the maximum target coverage
    print('Score: ', max(rewards))
    make_gif(folder=save_folder)
    return max(rewards)


# env_img: 3 * W * H, rgb format
def plot_ode_solving_process_on_image(ode_actions, env_img, img_id=0, save_folder='', save=False):
    '''
    visualize the generation process

    ode_actions:  intermediate variables during ODE solving process
    env_img: observation of PushT env
    img_id: the title or id current plot image, used for saving image
    save_folder: folder to save image
    save: whether to save or not
    '''
    for ode_step_id, ode_step_action in enumerate(ode_actions):
        ode_step_action = ode_step_action * 96 / 512
        plt.imshow(env_img)
        cmap = plt.cm.get_cmap('viridis')
        colors = np.linspace(0.1, 1, len(ode_step_action))
        plt.scatter(ode_step_action[..., 0], ode_step_action[..., 1], cmap=cmap, c=colors)
        plt.title('ODE solving at step ' + str(ode_step_id), fontsize=20)
        plt.xlim([-1, 96])
        plt.ylim([96, -1])
        plt.axis('off')
        if save:
            plt.savefig(save_folder + '/' + str(img_id) + '.png')
            img_id += 1
            plt.close()
        else:
            plt.draw()
            plt.pause(1.)
            plt.clf()
    return img_id


def plot_predicted_actions_series_on_iamge(action_series, env_img, save=False, img_id=0, save_folder=''):
    '''
    visualize the final generated action series on image

    action_series: generated action series by SRFMP
    env_img: observation of PushT env
    save: whether to save image
    img_id: id of current image, used for saving
    save_folder: folder to save image
    '''
    action_series = action_series * 96 / 512
    # fig = plt.figure()
    plt.imshow(env_img)
    cmap = plt.cm.get_cmap('viridis')
    colors = np.linspace(0.1, 1, len(action_series))
    plt.scatter(action_series[..., 0], action_series[..., 1], cmap=cmap, c=colors)
    plt.xlim([-1, 96])
    plt.ylim([96, -1])
    plt.axis('off')
    if save:
        plt.savefig(save_folder + '/' + str(img_id) + '.png')
        img_id += 1
        plt.close()
    else:
        plt.draw()


# todo
def make_gif(folder):
    import imageio
    from PIL import Image
    image_files = os.listdir(folder)
    sorted_image_files = sorted(image_files, key=lambda x: int(os.path.splitext(x)[0]))
    images = []
    for image_file in sorted_image_files:
        images.append(imageio.imread(folder + '/' + image_file))
    imageio.mimsave(folder + '/_gif.gif', images, fps=5, loop=0)


if __name__ == '__main__':
    # torch.manual_seed(3047)
    # Load config
    add_info = '_learntau_new_encoder'  #_learntau_new_encoder_lambdataux
    cfg = OmegaConf.load('refcond_srfm_euclidean_learntau_vision_pusht.yaml')
    # checkpoints_dir = "checkpoints/checkpoints_rfm_" + cfg.data + "_" + cfg.letter[0] + "_n" + str(cfg.n_pred)
    checkpoints_dir = "checkpoints/checkpoints_rfm_" + cfg.data + \
                      "_n" + str(cfg.n_pred) + "_r" + str(cfg.n_ref) + "_c" + str(cfg.n_cond) + "_w" + str(
        cfg.w_cond) + cfg.model_type + add_info
    checkpoints_dir = "checkpoints/checkpoints_rfm_" + cfg.data + \
                      "_n" + str(cfg.n_pred) + "_r" + str(cfg.n_ref) + "_c" + str(cfg.n_cond) + "_w" + str(
        cfg.w_cond) + cfg.model_type + add_info

    # Load dataset
    dataset, _ = _get_dataset(cfg)

    # Construct model
    model = SRFMVisionResnetTrajsModuleLearnTau(cfg)
    print(model)
    model.small_var = True
    best_checkpoint = glob(checkpoints_dir + "/**/epoch**.ckpt", recursive=True)[0]
    last_checkpoint = './' + checkpoints_dir + '/last.ckpt'
    # load model from checkpoint
    model = model.load_from_checkpoint(best_checkpoint, cfg=cfg)
    model.to(torch.device('cpu'))  # cuda  cpu

    # Training data statistics (min, max) for each dim
    stats = dataset.stats
    for i in range(1, 2):
        seed = i * 66
        ode_steps = 5
        sub_folder = 'ode' + str(ode_steps) + '_seed' + str(seed)
        if not os.path.exists(checkpoints_dir + '/plt_ode/' + sub_folder):
            os.mkdir(checkpoints_dir + '/plt_ode/' + sub_folder)
        inference(stats, model, obs_horizon=cfg.n_ref, pred_horizon=cfg.n_pred, action_horizon=int(8),
                  seed=seed, save_folder=checkpoints_dir + '/plt_ode/' + sub_folder, save=True, max_execution_steps=500,
                  ode_steps=ode_steps)

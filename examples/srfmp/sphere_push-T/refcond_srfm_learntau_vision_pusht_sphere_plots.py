import os.path
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm
import collections
from glob import glob
import cv2
import matplotlib

matplotlib.use('Agg')

from manifm.datasets import _get_dataset

from stable_flow.stable_model_vision_trajs_pl_learntau import SRFMVisionResnetTrajsModuleLearnTau
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv

import torchvision.transforms as Transform

from examples.datasets.create_pusht_sphere_image_dataset import project_sphere_action_back, \
    project_to_sphere_agent_pos, plt3d_sphere_project_img
import shutil


device = torch.device('cuda')


'''
test SRFMP Sphere PushT task
'''


def inference(model, obs_horizon, pred_horizon, action_horizon, seed=100000, crop=False, adp=False, ode_steps=10,
              scale=1.5, max_steps=500, new_unpack=False):
    '''once experiment of SRFMP on SPhere PushT task
                model: SRFMP model
                obs_horizon: observation horizon
                pred_horizon: prediction horizon
                action_horizon: execution horizon
                seed: seed for env initialization
                crop: whether to crop image
                adp: adaptive step size during ODE solving
                ode_steps: ODE solving steps
                scale: ratio of euclidean pusht env edge length and sphere to project
                max_steps: maximum roll outs
                new_unpack: sequential or parallel observation condition vector for different time frame
        '''
    if not os.path.exists(checkpoints_dir + '/automatic_dt/ode_step' + str(ode_steps)):
        os.mkdir(checkpoints_dir + '/automatic_dt/ode_step' + str(ode_steps))
    if not os.path.exists(checkpoints_dir + '/automatic_dt/ode_step' + '/seed' + str(seed) + '/sp_img'):
        os.mkdir(checkpoints_dir + '/automatic_dt/ode_step' + str(ode_steps) + '/seed' + str(seed))
        os.mkdir(checkpoints_dir + '/automatic_dt/ode_step' + str(ode_steps) + '/seed' + str(seed) + '/sp_img')
        os.mkdir(checkpoints_dir + '/automatic_dt/ode_step' + str(ode_steps) + '/seed' + str(seed) + '/sp_infer')

    # initial PushT env
    env = PushTImageEnv()
    # use a seed >200 to avoid initial states seen in the training dataset
    env.seed(seed)

    # get first observation
    # obs, info = env.reset()
    obs = env.reset()

    # keep a queue of last 2 steps of observations
    obs_deque = collections.deque(
        [obs] * obs_horizon, maxlen=obs_horizon)
    # save visualization and rewards
    cur_record_img = env.render(mode='rgb_array')
    imgs = [cur_record_img]
    rewards = list()
    done = False
    step_idx = 0

    with tqdm(total=max_steps, desc="Eval PushTStateEnv") as pbar:
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
                # save image for observation condition
                sp_cur_img = plt3d_sphere_project_img(255 * np.moveaxis(cur_img, 0, -1), scale=scale,
                                                      save_dir=checkpoints_dir + '/automatic_dt/ode_step' + str(ode_steps) + '/seed' + str(
                                                          seed) + '/sp_infer' + '/infer' + str(id))
                # sp_cur_img = sphere_project_img(255 * np.moveaxis(cur_img, 0, -1))
                sp_images[id] = np.moveaxis(sp_cur_img, -1, 0) / 255

            sp_images = torch.from_numpy(sp_images).to(device, dtype=torch.float32).unsqueeze(0).to(model.device)
            if crop:
                crop_transform = Transform.CenterCrop((84, 84))
                sp_images = crop_transform(sp_images)
            sp_agent_poses = torch.from_numpy(sp_agent_poses).to(device, dtype=torch.float32).to(model.device)
            if not new_unpack:
                sp_agent_poses = sp_agent_poses.reshape((B, model.n_ref * model.dim))

            with torch.no_grad():
                # get image features
                if new_unpack:
                    image_features = model.new_image_to_features(sp_images).squeeze(0)
                else:
                    image_features = model.image_to_features(sp_images)

                # concat with low-dim observations
                obs_features = torch.cat([image_features, sp_agent_poses], dim=-1)

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
            if 'euc' in cfg.data:
                naction = naction / np.linalg.norm(naction, axis=1).reshape((naction.shape[0], 1))
            action_pred = project_sphere_action_back(naction, r=1)
            action_pred = (action_pred / scale + 1) * 256
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
                cur_record_img = env.render(mode='rgb_array')
                imgs.append(cur_record_img)
                plt3d_sphere_project_img(cur_record_img, scale=scale,
                                         save_dir=checkpoints_dir + '/automatic_dt/ode_step' + str(ode_steps) + '/seed' + str(
                                             seed) + '/sp_img' + '/' + str(step_idx))

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

    # save the video and delete saved image
    make_video(checkpoints_dir + '/automatic_dt/ode_step' + str(ode_steps) + '/seed' + str(seed) + '/sp_img')
    shutil.rmtree(checkpoints_dir + '/automatic_dt/ode_step' + str(ode_steps) + '/seed' + str(seed) + '/sp_img')
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
    '''
    to make the file list as name order
    '''
    num_list = [int(file[:-4]) for file in file_list]
    sorted_lists = sorted(zip(num_list, file_list))
    _, file_list = zip(*sorted_lists)
    return file_list


if __name__ == '__main__':
    # Load config
    add_info = '_lambdax2.5'  # _learntau_new_encoder_lambdatau
    cfg = OmegaConf.load('refcond_srfm_sphere_learntau_vision_pusht.yaml')
    # checkpoints_dir = "checkpoints/checkpoints_rfm_" + cfg.data + "_" + cfg.letter[0] + "_n" + str(cfg.n_pred)
    checkpoints_dir = "checkpoints/checkpoints_rfm_" + cfg.data + \
                      "_n" + str(cfg.n_pred) + "_r" + str(cfg.n_ref) + "_c" + str(cfg.n_cond) + "_w" + str(
        cfg.w_cond) + cfg.model_type + add_info

    # Load dataset
    dataset, _ = _get_dataset(cfg)

    # Construct model
    model = SRFMVisionResnetTrajsModuleLearnTau(cfg)
    #print(model)
    model.small_var = False
    best_checkpoint = glob(checkpoints_dir + "/**/epoch**.ckpt", recursive=True)[0]
    last_checkpoint = './' + checkpoints_dir + '/last.ckpt'
    # load model from checkpoint
    model = model.load_from_checkpoint(best_checkpoint, cfg=cfg)
    model.to(torch.device('cuda'))  # cuda  cpu

    # Training data statistics (min, max) for each dim
    stats = dataset.stats

    # inference(stats, model, obs_horizon=cfg.n_ref, pred_horizon=cfg.n_pred, action_horizon=int(8),
    #           seed=66)
    if not os.path.exists(checkpoints_dir + '/automatic_dt'):
        os.mkdir(checkpoints_dir + '/automatic_dt')

    for ode_steps in [1]:
        reward_list = []
        # ode_steps = 1
        for seed in tqdm(range(1, 51)):
            reward = inference(model, obs_horizon=cfg.n_ref, pred_horizon=cfg.n_pred, action_horizon=int(8),
                               seed=seed * 66666, crop=True, adp=False, ode_steps=ode_steps, scale=1.5, new_unpack=False)
            reward_list.append(reward)
            # np.save(checkpoints_dir + '/automatic_dt/ode_step' + str(ode_steps) + '/reward_list' + '' + '.npy',
            #         reward_list)
            print('*********************reward mean ' + str(np.array(reward_list).mean()) + '************************')
        reward_list = np.array(reward_list)
        # np.save(checkpoints_dir + '/ode100_t1.5_seed1-50' + '/reward_list' + '' + '.npy', reward_list)
        print(reward_list)
        print(reward_list.mean())

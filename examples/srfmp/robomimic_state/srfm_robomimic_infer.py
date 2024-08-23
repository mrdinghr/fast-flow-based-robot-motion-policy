import numpy as np
from omegaconf import OmegaConf

from stable_flow.stable_model_trajs_robomimic_pl_learntau import SRFMRobomimicLTModule
from glob import glob
import torch
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import matplotlib

matplotlib.use('TkAgg')
from robomimic.config import config_factory
import robomimic.utils.obs_utils as ObsUtils

from robomimic.utils.train_utils import run_rollout
import matplotlib.pyplot as plt
import time
import collections
from tqdm import tqdm
import os


'''
test SRFMP on robomimic tasks with state-based observation
'''


# get observation condition vector
def get_ref(obs_deque):
    '''
    get observation condition vector
    '''
    pre_xref_object = obs_deque[0]['object']
    cur_xref_object = obs_deque[1]['object']
    xref_object = np.vstack((pre_xref_object, cur_xref_object))
    pre_xref_robo_pos = obs_deque[0]['robot0_eef_pos']
    cur_xref_robo_pos = obs_deque[1]['robot0_eef_pos']
    xref_robo_pos = np.vstack((pre_xref_robo_pos, cur_xref_robo_pos))
    pre_xref_robo_quat = obs_deque[0]['robot0_eef_quat']
    cur_xref_robo_quat = obs_deque[1]['robot0_eef_quat']
    xref_robo_quat = np.vstack((pre_xref_robo_quat, cur_xref_robo_quat))
    pre_xref_robo_gripper = obs_deque[0]['robot0_gripper_qpos']
    cur_xref_robo_gripper = obs_deque[1]['robot0_gripper_qpos']
    xref_robo_gripper = np.vstack((pre_xref_robo_gripper, cur_xref_robo_gripper))
    xref = np.concatenate((xref_object, xref_robo_pos, xref_robo_quat, xref_robo_gripper), axis=-1)
    return torch.from_numpy(xref)


# test with specific ODE steps
# visualize_onscreen: whether show the vidoe during test
def infer(model, ode_steps=11, max_steps=500, visualize_onscreen=False, execution_steps=8):
    '''once experiment: SRFMP on robomimic tasks with state-based observation

    model: SRFMP model
    ode_steps: ODE steps
    max_steps: maximum roll outs

    return:
    success or fail, recorded transition trajectory, recorded rotation trajectory
    '''
    # initialize robomimic env
    env_meta = FileUtils.get_env_metadata_from_dataset(
        '/home/dia1rng/robomimic/datasets/' + cfg.task + '/ph/low_dim_v141.hdf5')

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=env_meta["env_name"],
        render=visualize_onscreen,
        render_offscreen=False,
        use_image_obs=False,
    )
    # get initial observation
    obs = env.reset()
    cur_step = 0
    done = False
    obs_deque = collections.deque([obs] * cfg.n_ref, maxlen=cfg.n_ref)
    B = 1
    # record the trajectory seperately
    trans_trajectory = [obs['robot0_eef_pos']]
    rotation_trajectory = [obs['robot0_eef_quat']]
    while cur_step < max_steps and not done:
        # print('cur step ', cur_step)
        xref = get_ref(obs_deque)
        xref = xref.reshape((B, model.feature_dim * model.n_ref)).to(model.device).float()
        actions = model.sample_all(B, model.device, xref, ode_steps=ode_steps)
        actions = actions[-1][None][..., :model.dim * cfg.n_pred].reshape((model.n_pred, model.dim)).cpu().numpy()
        for action in actions[1: 1 + execution_steps]:
            obs, reward, done, _ = env.step(action)
            trans_trajectory.append(obs['robot0_eef_pos'])
            rotation_trajectory.append(obs['robot0_eef_quat'])
            # print('reward', reward)
            obs_deque.append(obs)
            cur_step += 1
            if visualize_onscreen:
                img = env.render(mode="rgb_array", camera_name="agentview", height=512, width=512)
                plt.imshow(img)
                plt.axis('off')
                plt.pause(0.02)
                plt.clf()
            # print(done)
            # early stop after success
            if reward > 0.4:
                return True, trans_trajectory, rotation_trajectory
    return False, trans_trajectory, rotation_trajectory


if __name__ == '__main__':
    # load config
    cfg = OmegaConf.load('refcond_srfm_robomimic.yaml')
    cfg.model_type = 'Unet'

    # set the config for Robomimic env
    config = config_factory(algo_name='bc')
    config.train.data = '/home/dia1rng/robomimic/datasets/' + cfg.task + '/ph/low_dim_v141.hdf5'
    config.train.batch_size = cfg.optim.batch_size
    config.experiment.validate = True
    config.train.hdf5_filter_key = "train"
    config.train.hdf5_validation_filter_key = "valid"
    config.train.frame_stack = cfg.n_ref
    config.train.seq_length = cfg.n_pred
    ObsUtils.initialize_obs_utils_with_config(config)
    model = SRFMRobomimicLTModule(cfg)

    # set checkpoints
    save_gap = 20
    save_num = 5
    if cfg.task == 'lift':
        max_steps = 300
    elif cfg.task == 'tool_hang':
        max_steps = 700
        save_gap = 20
    else:
        max_steps = 500
    add_info = '_saveevery' + str(save_gap) + '_total' + str(int(save_gap * save_num))

    checkpoints_dir = "checkpoints/checkpoints_rfm_" + cfg.data + \
                      "_n" + str(cfg.n_pred) + "_r" + str(cfg.n_ref) + "_c" + str(cfg.n_cond) + "_w" + str(
        cfg.w_cond) + cfg.model_type + cfg.task + add_info
    # best_checkpoint = glob(checkpoints_dir + "/**/epoch**.ckpt", recursive=True)[0]
    # last_checkpoint = './' + checkpoints_dir + '/last.ckpt'
    # model_checkpoint = './' + checkpoints_dir + '/model.ckpt'
    # model = model.load_from_checkpoint(last_checkpoint, cfg=cfg)
    # model.small_var = False
    # model.to(torch.device('cuda'))
    # # 1, 2, 3, 5, 10
    # for ode_steps in tqdm([1, 2, 3, 5, 10]):
    #     total_success_times = 0
    #     for seed in tqdm(range(1, 51)):
    #         np.random.seed(seed * 6)
    #         torch.manual_seed(seed * 6)
    #         success = infer(model, visualize_onscreen=False, ode_steps=ode_steps, max_steps=700)
    #         if success:
    #             total_success_times += 1
    #     print('success rate with ode steps ' + str(ode_steps) + ' ' + str(total_success_times / 50))
    #     np.save(checkpoints_dir + '/ode' + str(ode_steps) + '.npy', total_success_times / 50)

    # test with different checkpoints, ODE steps, seeds
    for epoch in range(5, 1 + save_num):
        if not os.path.exists(checkpoints_dir + '/epoch' + str(save_gap * epoch - 1)):
            os.mkdir(checkpoints_dir + '/epoch' + str(save_gap * epoch - 1))
        if save_gap * epoch - 1 < 10:
            cur_epoch_checkpoint = './' + checkpoints_dir + '/epoch-epoch=00' + str(
                save_gap * epoch - 1) + 'Unet' + cfg.task + '.ckpt'
        elif save_gap * epoch - 1 < 100:
            cur_epoch_checkpoint = './' + checkpoints_dir + '/epoch-epoch=0' + str(
                save_gap * epoch - 1) + 'Unet' + cfg.task + '.ckpt'
        else:
            cur_epoch_checkpoint = './' + checkpoints_dir + '/epoch-epoch=' + str(
                save_gap * epoch - 1) + 'Unet' + cfg.task + '.ckpt'
        model = SRFMRobomimicLTModule(cfg)
        model = model.load_from_checkpoint(cur_epoch_checkpoint, cfg=cfg)
        model.small_var = True
        model.to(torch.device('cuda'))
        # save_folder = checkpoints_dir + '/epoch' + str(save_gap * epoch - 1)
        # if not os.path.exists(save_folder):
        #     os.mkdir(save_folder)
        for ode_steps in tqdm([1, 2, 3, 5, 10]):
            save_folder = checkpoints_dir + '/epoch' + str(save_gap * epoch - 1) + '/ode' + str(ode_steps)
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
            total_success_times = 0
            for seed in tqdm(range(1, 51)):
                np.random.seed(seed * 6)
                torch.manual_seed(seed * 6)
                success, trans_traj, rotation_traj = infer(model, visualize_onscreen=False, ode_steps=ode_steps,
                                                           max_steps=max_steps)
                np.save(save_folder + '/seed' + str(seed * 6) + '_trajtrans.npy', trans_traj)
                np.save(save_folder + '/seed' + str(seed * 6) + '_trajrotate.npy', rotation_traj)
                if success:
                    total_success_times += 1
                    print('task ' + cfg.task + 'success at seed ' + str(seed * 6) + ' with ODE steps ' + str(ode_steps))
            print(
                'success rate with ode steps ' + str(ode_steps) + ' ' + str(total_success_times / 50) + ' epoch' + str(
                    epoch))
            np.save(checkpoints_dir + '/epoch' + str(epoch * save_gap - 1) + '_ode' + str(ode_steps) + '.npy',
                    total_success_times / 50)

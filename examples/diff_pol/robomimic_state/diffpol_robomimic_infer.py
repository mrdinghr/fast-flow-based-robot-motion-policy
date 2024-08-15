import numpy as np

from omegaconf import OmegaConf
import torch
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import matplotlib

matplotlib.use('TkAgg')
from robomimic.config import config_factory
import robomimic.utils.obs_utils as ObsUtils
import matplotlib.pyplot as plt
import collections
from tqdm import tqdm
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
import os


def get_ref(obs_deque):
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


def infer(nets, num_diffusion_steps=10, max_steps=500, visualize_onscreen=False, execution_steps=8):
    env_meta = FileUtils.get_env_metadata_from_dataset(
        '/home/dia1rng/robomimic/datasets/' + cfg.task + '/ph/low_dim_v141.hdf5')
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=env_meta["env_name"],
        render=visualize_onscreen,
        render_offscreen=False,
        use_image_obs=False,
    )
    obs = env.reset()
    cur_step = 0
    done = False
    obs_deque = collections.deque([obs] * cfg.n_ref, maxlen=cfg.n_ref)
    B = 1
    trans_trajectory = [obs['robot0_eef_pos']]
    rotation_trajectory = [obs['robot0_eef_quat']]
    while cur_step < max_steps and not done:
        xref = get_ref(obs_deque)
        xref = xref.reshape((B, obs_dim * cfg.n_ref)).to(device).float()
        actions = diffpol_action_generation(nets=nets, noise_scheduler=noise_scheduler, xref=xref, B=B,
                                            num_diffusion_steps=num_diffusion_steps)
        for action in actions[1:1 + execution_steps]:
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
                # plt.savefig('/home/dia1rng/Documents/thesis/first_script/img/robomimic/' + cfg.task + '.png',
                #             bbox_inches='tight')
                # plt.show()
            # print(done)
            if reward > 0.4:
                return True, trans_trajectory, rotation_trajectory
    return False, trans_trajectory, rotation_trajectory


def diffpol_action_generation(nets, noise_scheduler, xref, B, num_diffusion_steps):
    naction = torch.randn(
        (B, 16, action_dim), device=device)
    noise_scheduler.set_timesteps(num_diffusion_steps)
    for k in noise_scheduler.timesteps:
        # predict noise
        noise_pred = nets(
            sample=naction,
            timestep=k,
            global_cond=xref
        )

        # inverse diffusion step (remove noise)
        naction = noise_scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=naction
        ).prev_sample
    naction = naction.detach().to('cpu').numpy()
    return naction[0]


if __name__ == '__main__':
    cfg = OmegaConf.load('diffpol_robomimic.yaml')
    device = torch.device('cuda')
    config = config_factory(algo_name='bc')
    config.train.data = '/home/dia1rng/robomimic/datasets/' + cfg.task + '/ph/low_dim_v141.hdf5'
    config.train.batch_size = cfg.optim.batch_size
    config.experiment.validate = True
    config.train.hdf5_filter_key = "train"
    config.train.hdf5_validation_filter_key = "valid"
    config.train.frame_stack = cfg.n_ref
    config.train.seq_length = cfg.n_pred
    ObsUtils.initialize_obs_utils_with_config(config)
    action_dim = 7
    if cfg.task == 'can':
        ref_object_feature = 14
    elif cfg.task == 'lift':
        ref_object_feature = 10
    elif cfg.task == 'square':
        ref_object_feature = 14
    elif cfg.task == 'tool_hang':
        ref_object_feature = 44
    else:
        assert False, 'Wrong Task setting'
    obs_dim = ref_object_feature + 3 + 4 + 2
    obs_horizon = cfg.n_ref
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
    # noise_pred_net.load_state_dict(torch.load(cfg.task + '/' + cfg.diffusion_model + '_last.pt'))
    # noise_pred_net.to(device)
    #
    # for num_diffusion_steps in [1, 2, 3, 5, 10]:
    #     total_success_times = 0
    #     for seed in tqdm(range(1, 51)):
    #         np.random.seed(seed * 6)
    #         torch.manual_seed(seed * 6)
    #         success = infer(nets=noise_pred_net, num_diffusion_steps=num_diffusion_steps, visualize_onscreen=True)
    #         if success:
    #             total_success_times += 1
    #     print('success rate with ode steps ' + str(num_diffusion_steps) + ' ' + str(total_success_times / 50))
    #     # np.save(cfg.task + '/diffsteps' + str(num_diffusion_steps) + '.npy', total_success_times / 50)
    save_num = 5
    save_gap = 10
    total_epoch = int(save_gap * save_num)
    if cfg.task == 'tool_hang':
        max_steps = 700
        save_gap = 20
    elif cfg.task == 'lift':
        max_steps = 300
    else:
        max_steps = 500
    total_epoch = int(save_gap * save_num)
    for epoch_id in range(save_num, save_num + 1):
        model_folder = cfg.task + '_savegap' + str(save_gap) + '_total' + str(total_epoch)
        if not os.path.exists(model_folder + '/epoch' + str(save_gap * epoch_id - 1)):
            os.mkdir(model_folder + '/epoch' + str(save_gap * epoch_id - 1))
        noise_pred_net.load_state_dict(
            torch.load(model_folder + '/' + cfg.diffusion_model + '_epoch' + str(epoch_id * save_gap - 1) + '.pt'))
        noise_pred_net.to(device)

        for num_diffusion_steps in [1, 2, 5, 10]:
            save_folder = model_folder + '/epoch' + str(save_gap * epoch_id - 1) + '/step' + str(num_diffusion_steps)
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
            total_success_times = 0
            for seed in tqdm(range(1, 51)):
                np.random.seed(seed * 6)
                torch.manual_seed(seed * 6)
                success, trans_traj, rotation_traj = infer(nets=noise_pred_net, num_diffusion_steps=num_diffusion_steps,
                                                           visualize_onscreen=False, max_steps=max_steps)
                np.save(save_folder + '/seed' + str(seed * 6) + '_trajtrans.npy', trans_traj)
                np.save(save_folder + '/seed' + str(seed * 6) + '_trajrotate.npy', rotation_traj)
                if success:
                    total_success_times += 1
                    print('success at seed' + str(seed * 6) + ' with denoising step ' + str(
                        num_diffusion_steps) + ' epoch' + str(epoch_id))
            print('success rate with ode steps ' + str(num_diffusion_steps) + ' ' + str(
                total_success_times / 50) + ' epoch' + str(epoch_id))
            np.save(model_folder + '/epoch' + str(epoch_id * save_gap - 1) + 'diffsteps' + str(num_diffusion_steps) + '.npy', total_success_times / 50)

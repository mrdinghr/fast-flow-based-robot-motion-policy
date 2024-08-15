import os.path

from omegaconf import OmegaConf

from tami_clap_candidate.rpc_interface.rpc_interface import RPCInterface
from tami_clap_candidate.sensors.realsense import Preset, RealsenseRecorder
import time
import cv2
import torchvision.transforms as T
import torch
import numpy as np
import copy
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import collections

from tqdm import tqdm
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from manifm.vision.resnet_models import get_resnet, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from cup_on_plate_from_colab import DataUtils
import tempfile

GENERATE_TRAJECTORY = False
GRIPPER_OPEN_STATE = 0.02


def get_info(rpc):
    arm_state = rpc.get_robot_state()
    pose_loc = arm_state.pose.vec(quat_order="wxyz")
    gripper_state = arm_state.gripper[0]
    return pose_loc, gripper_state


def get_image(rs_recorder, transform_cam, device, show_on_screen=True, fix_img_list=None, grip_img_list=None):
    realsense_frames = rs_recorder.get_frame()
    to_plot = rs_recorder._visualize_frame(realsense_frames).copy()
    # if show_on_screen:
    #     plt.imshow(to_plot)

    image = np.stack(copy.deepcopy(to_plot), axis=0)  # stack to get image sequence [seq_len, H, W, BGR]
    image = image[:, :,
            ::-1].copy()  # reverse channel as re_recorder use cv2 -> BRG and we want RGB --> [seq_len, H, W, RGB]
    image = torch.as_tensor(image)
    image = image.float()
    image = image.moveaxis(-1, 0)  # [RGB, H, W]
    image = image / 255
    im_g = transform_cam(image[:, :, :640]).unsqueeze(0).to(
        device).float()  # unsqueeze to add batch dim -> [B, seq_len, RGB, H, W]
    # if show_on_screen:
    #     plt.imshow(im_g[0].moveaxis(0, -1).detach().cpu().numpy())
    if fix_img_list is not None:
        fix_img_list.append(image[:, :, 640:].cpu().numpy())
    im_f = transform_cam(image[:, :, 640:]).unsqueeze(0).to(
        device).float()  # unsqueeze to add batch dim -> [B, seq_len, RGB, H, W]
    if show_on_screen:
        plt.imshow(image.moveaxis(0, -1).detach().cpu().numpy())
        plt.draw()
        plt.pause(0.1)
        plt.clf()

    return im_f, im_g
    # SHAPE 1 * 3 * 480 * 640


def get_ref(state_list, grip_state_list, fix_img_list, grip_img_list, data_utils):
    pre_fix_img = fix_img_list[-2]
    cur_fix_img = fix_img_list[-1]
    pre_grip_img = grip_img_list[-2]
    cur_grip_img = grip_img_list[-1]  # 1 * rgb * W * H
    pre_fix_img_scalar = data_utils.image_to_features(pre_fix_img.unsqueeze(0))
    cur_fix_img_scalar = data_utils.image_to_features(cur_fix_img.unsqueeze(0))

    img_scalar = torch.cat((pre_fix_img_scalar, cur_fix_img_scalar), axis=1).to(device)

    xref_pos_quat = torch.tensor(state_list).to(device)
    xref_pos_quat = data_utils.normalize_pos_quat(xref_pos_quat).reshape((1, 14))
    print('normalize ref', xref_pos_quat)

    xref_gripper = torch.tensor(grip_state_list).unsqueeze(0).to(device)
    xref_gripper = data_utils.normalize_grip(xref_gripper)
    return torch.cat((img_scalar, xref_pos_quat, xref_gripper), axis=1).float()


def infer(nets, data_utils, rpc, cfg, execute_steps=8, obs_horizon=2, ctr_grip=False, video_name='', no_crop=True,
          num_diffusion_iters=10, crop_percent=0.2, save_folder=''):
    step = 0
    # resize Height * Width
    if no_crop:
        transform_cam = T.Compose([T.Resize((cfg.image_height, cfg.image_width), antialias=None), ])
    else:
        crop_height = int(cfg.image_height * (1 - crop_percent))
        crop_width = int(cfg.image_width * (1 - crop_percent))
        transform_cam = T.Compose(
            [T.Resize((cfg.image_height, cfg.image_width), antialias=None), T.CenterCrop((crop_height, crop_width)), ])
    rs_recorder = RealsenseRecorder(
        height=480,
        width=640,
        fps=30,
        record_depth=False,
        depth_unit=0.001,
        preset=Preset.HighAccuracy,
        memory_first=True,
    )

    pose_loc, gripper_state = get_info(rpc)
    fix_img, gripper_image = get_image(rs_recorder=rs_recorder, transform_cam=transform_cam, device=device)

    state_deque = collections.deque([pose_loc] * obs_horizon, maxlen=obs_horizon)
    fix_img_deque = collections.deque([fix_img] * obs_horizon, maxlen=obs_horizon)
    grip_img_deque = collections.deque([gripper_image] * obs_horizon, maxlen=obs_horizon)
    grip_state_deque = collections.deque([gripper_state] * obs_horizon, maxlen=obs_horizon)
    real_state_list = []
    predict_action_list = []
    # state_list = [pose_loc, pose_loc]
    fix_img_list = []
    grip_img_list = []
    infer_time_list = []
    # grip_img_list = [gripper_image, gripper_image]
    # grip_state_list = [gripper_state, gripper_state]
    B = 1
    infer_times = 0
    while True:
        print('**********infer times ' + str(infer_times) + '*********************')
        infer_times += 1
        obs_cond = get_ref(state_deque, grip_state_deque, fix_img_deque, grip_img_deque, data_utils)
        noise_scheduler.set_timesteps(num_diffusion_iters)
        # get actions series
        start_time = time.time()
        noisy_action = torch.randn(
            (B, 16, action_dim), device=device)
        naction = noisy_action
        with torch.no_grad():
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
        print('inference uses ' + str(time.time() - start_time))
        infer_time_list.append(time.time() - start_time)
        actions = naction[0]
        if cfg.normalize_pos_quat:
            actions[..., :-1] = data_utils.denormalize_pos_quat(actions[..., :-1])
            actions[..., -1] = data_utils.denormalize_grip(actions[..., -1])
        actions = actions.cpu().numpy()
        # print('state deque', state_deque)
        # print('pred action', actions)
        for id, action in enumerate(actions[1:1 + execute_steps]):
            predict_action_list.append(action)
            # rpc.goto_cartesian_pose_nonblocking(action[:3], action[3:7], GENERATE_TRAJECTORY)
            # rpc.goto_cartesian_pose_nonblocking(action[:3], action[3:7] / np.linalg.norm(action[3:7]), GENERATE_TRAJECTORY)
            if ctr_grip:
                if gripper_state >= GRIPPER_OPEN_STATE:  # if open
                    if action[-1] < GRIPPER_OPEN_STATE:
                        rpc.close_gripper()
                elif gripper_state < GRIPPER_OPEN_STATE:  # if close
                    if action[-1] > GRIPPER_OPEN_STATE:
                        rpc.open_gripper()
            # print('action', action)
            time.sleep(0.1)
            pose_loc, gripper_state = get_info(rpc)
            real_state_list.append(pose_loc)
            # print('cur state', pose_loc)
            state_deque.append(pose_loc)
            grip_state_deque.append(gripper_state)

            fix_img, gripper_image = get_image(rs_recorder=rs_recorder, transform_cam=transform_cam,
                                               device=device, fix_img_list=fix_img_list,
                                               grip_img_list=grip_img_list)
            fix_img_deque.append(fix_img)
            grip_img_deque.append(gripper_image)
            # fix_img_list.append(fix_img.cpu().numpy())
            step += 1
            # save_array_safe(fix_img_list, save_folder + '/img_' + video_name + '_exc' + str(
            #     execute_steps) + '_denoisestep' + str(num_diffusion_iters) + '.npy')
            # save_array_safe(real_state_list, save_folder + '/state_' + video_name + '_exc' + str(
            #     execute_steps) + '_denoisestep' + str(num_diffusion_iters) + '.npy')
            # save_array_safe(predict_action_list, save_folder + '/action_' + video_name + '_exc' + str(
            #     execute_steps) + '_denoisestep' + str(num_diffusion_iters) + '.npy')
            # save_array_safe(infer_time_list, save_folder + '/time_' + video_name + '_exc' + str(
            #     execute_steps) + '_denoisestep' + str(num_diffusion_iters) + '.npy')
            save_array_safe(infer_time_list, save_folder + '/time_step' + str(num_diffusion_iters) + '.npy')
            # np.save('DDIM' + '/block/img' + video_name, fix_img_list)
            # np.save('DDIM' + '/block/state' + video_name, real_state_list)

    # vwrite('infer.mp4', fix_img_list)


def save_array_safe(arr, filename):
    """Saves an array to a file with atomic writes.

    Args:
      arr: The NumPy array to save.
      filename: The final filename to save the array to.
    """
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        np.save(temp_file, arr)  # Save to temporary file
        temp_file.close()  # Close the temporary file (important)

    # Rename the temporary file to the final filename (atomic rename)
    import os
    os.replace(temp_file.name, filename)


if __name__ == '__main__':
    device = torch.device('cuda')
    cfg = OmegaConf.load('diffpol_cuponplate.yaml')
    add_info = ''

    # load diffusion model
    action_dim = 3 + 4 + 1
    vision_feature_dim = 512 + 8
    obs_horizon = 2
    n_ref = cfg.n_ref
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=n_ref * vision_feature_dim,
        down_dims=[128, 256, 512],
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
    vision_encoder = get_resnet('resnet18')
    vision_encoder = replace_bn_with_gn(vision_encoder)
    nets = torch.nn.ModuleDict({'vision_encoder': vision_encoder, 'noise_pred_net': noise_pred_net})
    nets.load_state_dict(torch.load('DDIM/DDIM.pt'))
    nets.to(device)
    data_utils = DataUtils(cfg=cfg, nets=nets)

    rpc = RPCInterface("10.87.170.254")  # 10.87.172.60

    for _ in range(5):
        rpc.open_gripper()

    success = rpc.goto_home_joint()
    # target_pose_1 = np.array([0.47, 0.05, 0.24, 0.0, 1.0, 0.0, 0.0])
    target_pose_1 = np.array([0.47, 0.10, 0.30, 0.0, 1.0, 0.0, 0.0])
    # target_pose_1[2] = 0.2
    success = rpc.activate_cartesian_nonblocking_controller()

    rpc.goto_cartesian_pose_blocking(target_pose_1[:3], target_pose_1[3:], True)
    time.sleep(0.5)
    if success:
        print("controller activated successfully")
    else:
        print("control activation failed :(")

    num_diffusion_iters = 2
    save_folder = 'DDIM/time'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    infer(nets=nets, data_utils=data_utils, rpc=rpc, cfg=cfg, execute_steps=8, ctr_grip=True,
          video_name='7', no_crop=False, crop_percent=0.1, num_diffusion_iters=num_diffusion_iters,
          save_folder=save_folder)
    # infer_with_dataloader(model=model, rpc=rpc, cfg=cfg, data_loader=val_loader, pos_use_loadar=False)

    # reset the robot arm
    joint_config_1 = [0.0, -0.7, 0.035, -2.45, 0.0, 1.81, 0.73]
    success = rpc.goto_joint_position_blocking(joint_config_1, True)

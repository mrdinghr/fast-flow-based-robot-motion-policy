import os
from stable_flow.stable_model_vision_dishgrasp_pl_learntau import SRFMVisionResnetTrajsModuleLearnTau
from omegaconf import OmegaConf
from glob import glob

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

from dummy_robot_arm1 import get_debug_loaders
from types import SimpleNamespace
from tqdm import tqdm
import tempfile

GENERATE_TRAJECTORY = False
GRIPPER_OPEN_STATE = 0.02


'''
Test SRFMP on real robot Pick Place task
'''


# get robot end effector state
def get_info(rpc):
    '''
    get robot end effector state from rpc interface
    '''
    arm_state = rpc.get_robot_state()
    pose_loc = arm_state.pose.vec(quat_order="wxyz")
    gripper_state = arm_state.gripper[0]
    return pose_loc, gripper_state


# image observation vector
def get_image(rs_recorder, transform_cam, device, show_on_screen=True, fix_img_list=None, use_grip_img=False,
              grip_img_list=None):
    '''
    get image from realsense interface
    '''
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
        if use_grip_img:
            grip_img_list.append(image[:, :, :640].cpu().numpy())
    im_f = transform_cam(image[:, :, 640:]).unsqueeze(0).to(
        device).float()  # unsqueeze to add batch dim -> [B, seq_len, RGB, H, W]
    if show_on_screen:
        if not model.use_gripper_img:
            plt.imshow(im_f[0].moveaxis(0, -1).detach().cpu().numpy())
            plt.draw()
            plt.pause(0.2)
            plt.clf()
        else:
            plt.imshow(image.moveaxis(0, -1).detach().cpu().numpy())
            plt.draw()
            plt.pause(0.2)
            plt.clf()

    return im_f, im_g
    # SHAPE 1 * 3 * 480 * 640


# observation vector
def get_ref(state_list, grip_state_list, fix_img_list, grip_img_list, model):
    '''
    get observation condition vector

    state_list: list of past 2 frame end effector state
    grip_sate_list: list of past 2 frame gripper state
    fix_img_list: list of past 2 framw over-the-shoulder camera observation
    grip_img_list: list of past 2 fram in-hnad camera observation
    use_imgae: stae or vision bsed observation
    model: SRFMP model
    '''
    pre_fix_img = fix_img_list[-2]
    cur_fix_img = fix_img_list[-1]
    pre_grip_img = grip_img_list[-2]
    cur_grip_img = grip_img_list[-1]  # 1 * rgb * W * H
    pre_fix_img_scalar = model.image_to_features(pre_fix_img.unsqueeze(0))
    cur_fix_img_scalar = model.image_to_features(cur_fix_img.unsqueeze(0))

    if model.use_gripper_img:
        pre_grip_img_scalar = model.grip_image_to_features(pre_grip_img.unsqueeze(0))
        cur_grip_img_scalar = model.grip_image_to_features(cur_grip_img.unsqueeze(0))
        img_scalar = torch.cat((pre_fix_img_scalar, pre_grip_img_scalar, cur_fix_img_scalar, cur_grip_img_scalar),
                               axis=1)
    else:
        img_scalar = torch.cat((pre_fix_img_scalar, cur_fix_img_scalar), axis=1)
    if model.normalize:
        xref_pos_quat = torch.tensor(state_list).to(model.device)
        xref_pos_quat = model.normalize_pos_quat(xref_pos_quat).reshape((1, 14))
        print('normalize ref', xref_pos_quat)
    else:
        xref_pos_quat = torch.tensor(state_list).reshape((1, 14)).to(model.device)
    xref_gripper = torch.tensor(grip_state_list).unsqueeze(0).to(model.device)
    if model.normalize:
        xref_gripper = model.normalize_grip(xref_gripper)
    return torch.cat((img_scalar, xref_pos_quat, xref_gripper), axis=1).float()


# once experiment with SRFMP on task Pick Place
def infer(model, rpc, cfg, execute_steps=8, obs_horizon=2, ctr_grip=False, video_name='', no_crop=True,
          adp=False, crop_percent=0.2, save_folder='', ode_num=5):
    '''once experiment of SRFMP on Pick Place real robot task
    model: SRFMP model
    rpc: interface with robot
    cfg: config file
    execute_steps: execution horizon
    obs_horizon: observation horizon
    ctr_grip: whether to control gripper or not
    video_name: name of saved video
    np_crop: whether to crop image or not
    adp: adaptive step size for ODE solving
    crop_percent: percentage of crop
    save_folder: folder to save all experiment data
    ode_num: ODE solving steps
    '''

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
    fix_img, gripper_image = get_image(rs_recorder=rs_recorder, transform_cam=transform_cam, device=model.device)

    state_deque = collections.deque([pose_loc] * obs_horizon, maxlen=obs_horizon)
    fix_img_deque = collections.deque([fix_img] * obs_horizon, maxlen=obs_horizon)
    grip_img_deque = collections.deque([gripper_image] * obs_horizon, maxlen=obs_horizon)
    grip_state_deque = collections.deque([gripper_state] * obs_horizon, maxlen=obs_horizon)
    # state_list = [pose_loc, pose_loc]
    real_state_list = []
    predict_action_list = []
    fix_img_list = []
    grip_img_list = []
    infer_time_list = []
    if model.use_gripper_img:
        grip_img_video_name = 'grip_' + video_name
    # grip_img_list = [gripper_image, gripper_image]
    # grip_state_list = [gripper_state, gripper_state]
    infer_total_times = 0
    while True:
        print('************ total infer ' + str(infer_total_times) + '*****************')
        infer_total_times += 1
        xref = get_ref(state_deque, grip_state_deque, fix_img_deque, grip_img_deque, model)
        # get actions series
        start_time = time.time()
        actions = model.sample_all(n_samples=1, device=model.device, xref=xref, different_sample=True, ode_steps=ode_num,
                                   adp=adp)
        infer_time_list.append(time.time() - start_time)
        print('inference uses ' + str(time.time() - start_time))
        # print('tau series ', actions[:, 0, -1])
        actions = actions[-1, 0, :-1].reshape((cfg.n_pred, model.dim))
        if cfg.normalize_pos_quat:
            actions[..., :-1] = model.denormalize_pos_quat(actions[..., :-1])
            actions[..., -1] = model.denormalize_grip(actions[..., -1])
        actions = actions.cpu().numpy()
        # print('state deque', state_deque)
        # print('pred action', actions[..., :3])
        for id, action in enumerate(actions[1:1 + execute_steps]):
            predict_action_list.append(action)
            # rpc.goto_cartesian_pose_nonblocking(action[:3], action[3:7], GENERATE_TRAJECTORY)
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
                                               device=model.device, fix_img_list=fix_img_list,
                                               use_grip_img=model.use_gripper_img, grip_img_list=grip_img_list)
            fix_img_deque.append(fix_img)
            grip_img_deque.append(gripper_image)
            # fix_img_list.append(fix_img.cpu().numpy())
            step += 1
            # save_array_safe(fix_img_list, save_folder + '/img' + video_name + '_exc' + str(execute_steps) + '.npy')
            # save_array_safe(real_state_list,
            #                 save_folder + '/state' + video_name + '_exc' + str(execute_steps) + '.npy')
            # save_array_safe(predict_action_list,
            #                 save_folder + '/action' + video_name + '_exc' + str(execute_steps) + '.npy')
            # save_array_safe(infer_time_list, save_folder + '/time_' + video_name + '.npy')
            save_array_safe(infer_time_list, save_folder + '/time_step' + str(ode_num))

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

    os.replace(temp_file.name, filename)


def infer_with_dataloader(model, rpc, cfg, execute_steps=8, obs_horizon=2, data_loader=None, pos_use_loadar=False):
    '''experiment with SRFMP on real robot but get observation from dataloadar

           model: RFMP model
           rpc: interface with robot
           cfg: config file
           execute_steps: execution horizon
           obs_horizon: observation horizon
           data_loadar: data loadar
           pos_use_loadar: whether to use state information from dataloadar or read from real robot
           '''
    crop_height = int(cfg.image_height * 0.9)
    crop_width = int(cfg.image_width * 0.9)
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

    fix_img_real_list = []
    real_state = []
    demo_state = []
    pred_state = []
    for i, batch in tqdm(enumerate(data_loader)):
        if i % execute_steps == 0:
            _, xref, xcond, _ = model.unpack_predictions_reference_conditioning_samples(batch)
            start_time = time.time()
            actions = model.sample_all(1, model.device, xref=xref, xcond=xcond, different_sample=True, ode_steps=10,
                                       adp=False)
            print('tau series ', actions[:, 0, -1])
            actions = actions[-1, 0, :-1].reshape((cfg.n_pred, model.dim))
            if cfg.normalize_pos_quat:
                actions[..., :-1] = model.denormalize_pos_quat(actions[..., :-1])
                actions[..., -1] = model.denormalize_grip(actions[..., -1])
            for idx, action in enumerate(actions[1: 1 + execute_steps]):
                fix_img = batch['observation']['v_fix']
                plt.imshow(fix_img[0, -1].moveaxis(0, -1))
                plt.draw()
                plt.pause(0.2)
                plt.clf()
                rpc.goto_cartesian_pose_nonblocking(action[:3], action[3:7], GENERATE_TRAJECTORY)


if __name__ == '__main__':
    # specific info for used model
    add_info = ('_tauencode_lambda25_crop1')
    # _fewepochs_smallertaunet_uncorrelattauvf
    cfg = OmegaConf.load('srfm_cuponplate.yaml')
    checkpoints_dir = "checkpoints/checkpoints_rfm_" + cfg.data + \
                      "_n" + str(cfg.n_pred) + "_r" + str(cfg.n_ref) + "_c" + str(cfg.n_cond) + "_w" + str(
        cfg.w_cond) + cfg.model_type + add_info
    # construct model
    model = SRFMVisionResnetTrajsModuleLearnTau(cfg)

    best_checkpoint = glob(checkpoints_dir + "/**/epoch**.ckpt", recursive=True)[0]
    last_checkpoint = './' + checkpoints_dir + '/last.ckpt'
    model_checkpoint = './' + checkpoints_dir + '/model.ckpt'
    # load model from checkpoints
    model = model.load_from_checkpoint(best_checkpoint, cfg=cfg)
    model.to(torch.device('cuda'))
    model.small_var = True
    print('normalize', model.normalize)
    print('use gripper img ', model.use_gripper_img)

    # initialize robot interface
    rpc = RPCInterface("10.87.170.254")  # 10.87.172.60

    data_folder_path = '/home/dia1rng/hackathon/hachaton_cuponboard_new_dataset/cuponplate1_robot_demos'
    args = SimpleNamespace()

    args.ablation = 'vf_vg'
    args.num_stack = 2
    args.frameskip = 1
    args.no_crop = False
    args.crop_percent = 0.3
    args.resized_height_v = cfg.image_height
    args.resized_width_v = cfg.image_width
    args.len_lb = cfg.n_pred - 1
    args.sampling_time = 250
    args.norm = False
    args.source = False
    train_loader, val_loader, _ = get_debug_loaders(batch_size=1, args=args, data_folder=data_folder_path,
                                                    drop_last=True)
    # for _ in range(5):
    #     rpc.open_gripper()

    success = rpc.goto_home_joint()
    target_pose_1 = np.array([0.43, 0.045, 0.30, 0.0, 1.0, 0.0, 0.0])
    target_pose_1 = np.array([0.42, 0.08, 0.30, 0.0, 1.0, 0.0, 0.0])
    # target_pose_1 = np.array([0.4, 0.0, 0.30, 0.0, 1.0, 0.0, 0.0])
    target_pose_1[0] -= 0.03
    success = rpc.activate_cartesian_nonblocking_controller()

    rpc.goto_cartesian_pose_blocking(target_pose_1[:3], target_pose_1[3:], True)
    # time.sleep(0.5)
    if success:
        print("controller activated successfully")
    else:
        print("control activation failed :(")

    ode_num = 5
    save_folder = checkpoints_dir + '/time'

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    infer(model, rpc, cfg, execute_steps=8, ctr_grip=True, video_name='10',
          no_crop=False, adp=False, crop_percent=0.1, ode_num=ode_num, save_folder=save_folder)
    # infer_with_dataloader(model=model, rpc=rpc, cfg=cfg, data_loader=val_loader, pos_use_loadar=True)

    # reset the robot arm
    joint_config_1 = [0.0, -0.7, 0.035, -2.45, 0.0, 1.81, 0.73]
    success = rpc.goto_joint_position_blocking(joint_config_1, True)

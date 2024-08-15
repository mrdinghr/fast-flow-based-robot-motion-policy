import os.path
import sys

sys.path.append('/home/dia1rng/hackathon/flow-matching-policies/stable_flow')
from stable_model_vision_rotatemug_pl_learntau import SRFMRotateMugVisionResnetTrajsModuleLearnTau
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

from types import SimpleNamespace
from tqdm import tqdm
import tempfile

GENERATE_TRAJECTORY = False
GRIPPER_OPEN_STATE = 0.02


def get_info(rpc):
    arm_state = rpc.get_robot_state()
    pose_loc = arm_state.pose.vec(quat_order="wxyz")
    gripper_state = arm_state.gripper[0]
    return pose_loc, gripper_state


def get_image(rs_recorder, transform_cam, device, show_on_screen=True, fix_img_list=None, use_grip_img=False,
              grip_img_list=None):
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


def get_ref(state_list, grip_state_list, fix_img_list, grip_img_list, model):
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


def infer(model, rpc, cfg, execute_steps=8, obs_horizon=2, ctr_grip=False, video_name='', no_crop=True,
          adp=False, ode_steps=10, crop_percent=0.2, save_folder=''):
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
    print('initial state ', pose_loc)
    fix_img, gripper_image = get_image(rs_recorder=rs_recorder, transform_cam=transform_cam, device=model.device)

    state_deque = collections.deque([pose_loc] * obs_horizon, maxlen=obs_horizon)
    fix_img_deque = collections.deque([fix_img] * obs_horizon, maxlen=obs_horizon)
    grip_img_deque = collections.deque([gripper_image] * obs_horizon, maxlen=obs_horizon)
    grip_state_deque = collections.deque([gripper_state] * obs_horizon, maxlen=obs_horizon)
    # state_list = [pose_loc, pose_loc]
    real_state_list = []
    fix_img_list = []
    grip_img_list = []
    predict_action_list = []
    infer_time_list = []
    if model.use_gripper_img:
        grip_img_video_name = 'grip_' + video_name
    # grip_img_list = [gripper_image, gripper_image]
    # grip_state_list = [gripper_state, gripper_state]

    while True:
        xref = get_ref(state_deque, grip_state_deque, fix_img_deque, grip_img_deque, model)
        # get actions series
        start_time = time.time()
        actions = model.sample_all(n_samples=1, device=model.device, xref=xref, different_sample=True,
                                   ode_steps=ode_steps, adp=adp)
        infer_time_list.append(time.time() - start_time)
        print('inference uses ' + str(time.time() - start_time))
        print('tau series ', actions[:, 0, -1])
        actions = actions[-1, 0, :-1].reshape((cfg.n_pred, model.dim))
        if cfg.normalize_pos_quat:
            actions[..., :-1] = model.denormalize_pos_quat(actions[..., :-1])
            actions[..., -1] = model.denormalize_grip(actions[..., -1])
        actions = actions.cpu().numpy()
        print('pred action', actions[..., :])
        for id, action in enumerate(actions[1:1 + execute_steps]):
            print('execute action ', action)
            predict_action_list.append(action)
            rpc.goto_cartesian_pose_nonblocking(action[:3], action[3:7], GENERATE_TRAJECTORY)
            # rpc.goto_cartesian_pose_blocking(action[:3], action[3:7], GENERATE_TRAJECTORY)
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
            save_array_safe(fix_img_list, save_folder + '/img' + video_name)
            save_array_safe(real_state_list, save_folder + '/state' + video_name)
            save_array_safe(predict_action_list, save_folder + '/action' + video_name)
            save_array_safe(infer_time_list, save_folder + '/time' + video_name)
            # if model.use_gripper_img:
            #     np.save(checkpoints_dir + '/' + grip_img_video_name, grip_img_list)
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


def infer_with_dataloader(model, rpc, cfg, execute_steps=8, obs_horizon=2, data_loader=None, pos_use_loadar=False):
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
    add_info = 'lambdataux25_harddata'
    # _fewepochs_smallertaunet_uncorrelattauvf
    cfg = OmegaConf.load('srfm_rotate_mug.yaml')
    checkpoints_dir = "checkpoints/checkpoints_rfm_" + cfg.data + \
                      "_n" + str(cfg.n_pred) + "_r" + str(cfg.n_ref) + "_c" + str(cfg.n_cond) + "_w" + str(
        cfg.w_cond) + cfg.model_type + add_info
    # model = ManifoldVisionTrajectoriesDishGraspFMLitModule(cfg)
    model = SRFMRotateMugVisionResnetTrajsModuleLearnTau(cfg)

    best_checkpoint = glob(checkpoints_dir + "/**/epoch**.ckpt", recursive=True)[0]
    last_checkpoint = './' + checkpoints_dir + '/last.ckpt'
    model_checkpoint = './' + checkpoints_dir + '/model.ckpt'
    model = model.load_from_checkpoint(best_checkpoint, cfg=cfg)
    model.to(torch.device('cuda'))
    model.small_var = True
    print('normalize', model.normalize)
    print('use gripper img ', model.use_gripper_img)

    rpc = RPCInterface("10.87.170.254")  # 10.87.172.60

    # data_folder_path = '/home/dia1rng/hackathon/hachaton_cuponboard_new_dataset/cuponplate1_robot_demos'
    # args = SimpleNamespace()
    #
    # args.ablation = 'vf_vg'
    # args.num_stack = 2
    # args.frameskip = 1
    # args.no_crop = False
    # args.crop_percent = 0.3
    # args.resized_height_v = cfg.image_height
    # args.resized_width_v = cfg.image_width
    # args.len_lb = cfg.n_pred - 1
    # args.sampling_time = 250
    # args.norm = False
    # args.source = False
    # train_loader, val_loader, _ = get_debug_loaders(batch_size=1, args=args, data_folder=data_folder_path,
    #                                                 drop_last=True)
    for _ in range(5):
        rpc.open_gripper()

    success = rpc.goto_home_joint()
    target_pose_1 = np.array([0.43, 0.045, 0.30, 0.0, 1.0, 0.0, 0.0])
    target_pose_1 = np.array([0.37888516710211556,
                              0.00510292887852903276,
                              0.35864418892044,
                              0.0030079931156787435,
                              -0.9972750881656733,
                              0.053389844167177965,
                              0.05082199367762835])
    # target_pose_1 = np.array([
    #     0.38220130741491637,
    #     0.010190241775998542,
    #     0.36850870711719125,
    #     0.01236478116009615,
    #     -0.9978602092071823,
    #     0.049083418598210055,
    #     0.04138759580567998
    # ])
    # target_pose_1 = np.array([0.4, 0.0, 0.30, 0.0, 1.0, 0.0, 0.0])
    # target_pose_1[2] += 0.05
    success = rpc.activate_cartesian_nonblocking_controller()

    rpc.goto_cartesian_pose_blocking(target_pose_1[:3], target_pose_1[3:], True)
    time.sleep(0.5)
    if success:
        print("controller activated successfully")
    else:
        print("control activation failed :(")

    execution_steps = 16
    ode_steps = 1
    save_folder = checkpoints_dir + '/ode' + str(ode_steps)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    infer(model, rpc, cfg, execute_steps=execution_steps, ctr_grip=True,
          video_name='' + '2' + '_exc' + str(execution_steps) + '.npy',
          no_crop=False, adp=False, ode_steps=ode_steps + 1, crop_percent=0.2, save_folder=save_folder)
    # infer_with_dataloader(model=model, rpc=rpc, cfg=cfg, data_loader=val_loader, pos_use_loadar=True)

    # reset the robot arm
    joint_config_1 = [0.0, -0.7, 0.035, -2.45, 0.0, 1.81, 0.73]
    success = rpc.goto_joint_position_blocking(joint_config_1, True)

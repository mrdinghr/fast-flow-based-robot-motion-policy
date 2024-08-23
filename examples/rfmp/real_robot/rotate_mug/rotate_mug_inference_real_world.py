from manifm.model_trajs_vision_rotate_mug_pl import ManifoldVisionTrajectoriesMugRotateFMLitModule
from omegaconf import OmegaConf
from glob import glob
from tami_clap_candidate.rpc_interface.rpc_interface import RPCInterface
from tami_clap_candidate.sensors.realsense import Preset, RealsenseRecorder
import time

import torchvision.transforms as T
import torch
import numpy as np
import copy
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import collections
from tqdm import tqdm

import tempfile
import os

GENERATE_TRAJECTORY = False
GRIPPER_OPEN_STATE = 0.02


'''
script for testing RFMP on task Rotate Mug on real robot
'''


def get_info(rpc):
    '''
    get end effector state from rpc interface
    '''
    arm_state = rpc.get_robot_state()
    pose_loc = arm_state.pose.vec(quat_order="wxyz")
    pose_loc[3:7] *= np.sign(pose_loc[4])
    gripper_state = arm_state.gripper[0]
    return pose_loc, gripper_state


def get_image(rs_recorder, transform_cam, device, show_on_screen=True, fix_img_list=None, use_grip_img=False,
              grip_img_list=None):
    '''
    get image observation from realsense interface
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
        plt.imshow(image.moveaxis(0, -1).detach().cpu().numpy())
        plt.draw()
        plt.pause(0.1)
        plt.clf()

    return im_f, im_g
    # SHAPE 1 * 3 * 480 * 640


def get_ref(state_list, grip_state_list, fix_img_list, grip_img_list, model, use_image=False):
    '''
    get observation condition vector

    state_list: list of past 2 frame end effector state
    grip_sate_list: list of past 2 frame gripper state
    fix_img_list: list of past 2 framw over-the-shoulder camera observation
    grip_img_list: list of past 2 fram in-hnad camera observation
    use_imgae: stae or vision bsed observation
    '''
    pre_fix_img = fix_img_list[-2]
    cur_fix_img = fix_img_list[-1]
    pre_grip_img = grip_img_list[-2]
    cur_grip_img = grip_img_list[-1]  # 1 * rgb * W * H
    if use_image:
        pre_fix_img_scalar = model.image_to_features(pre_fix_img.unsqueeze(0))
        cur_fix_img_scalar = model.image_to_features(cur_fix_img.unsqueeze(0))
        img_scalar = torch.cat((pre_fix_img_scalar, cur_fix_img_scalar), axis=1)
    print('reference state ', state_list)
    if model.normalize:
        xref_pos_quat = torch.tensor(state_list).to(model.device)
        xref_pos_quat = model.normalize_pos_quat(xref_pos_quat).reshape((1, 14))
        # print('normalize ref', xref_pos_quat)
    else:
        xref_pos_quat = torch.tensor(state_list).reshape((1, 14)).to(model.device)
    xref_gripper = torch.tensor(grip_state_list).unsqueeze(0).to(model.device)
    if model.normalize:
        xref_gripper = model.normalize_grip(xref_gripper)
    if not use_image:
        return torch.cat((xref_pos_quat, xref_gripper), axis=1).float()
    else:
        return torch.cat((img_scalar, xref_pos_quat, xref_gripper), axis=1).float()


def get_ref_infer_with_dataloader(state_list, grip_state_list, fix_img_list, grip_img_list, model, use_s_loadar=False):
    '''get obseravtion condition vector when getting observation from dataloader
        state_list: list containing past 2 frame end effector state
        grip_state_list: list containing past 2 frame gripper state
        fix_img_list: containing past 2 frame over-the-shoulder camera observation
        grip_img_list: containing past 2 frame
        model: RFMP model
        use-s_loadar: whether to use state information from dataloadar or read from real robot
        '''
    pre_fix_img = fix_img_list[-2].to(model.device)
    cur_fix_img = fix_img_list[-1].to(model.device)
    pre_grip_img = grip_img_list[-2].to(model.device)
    cur_grip_img = grip_img_list[-1].to(model.device)  # 1 * rgb * W * H
    pre_fix_img_scalar = model.image_to_features(pre_fix_img)
    cur_fix_img_scalar = model.image_to_features(cur_fix_img)
    pre_grip_img_scalar = model.image_to_features(pre_grip_img)
    cur_grip_img_scalar = model.image_to_features(cur_grip_img)
    if model.use_gripper_img:
        img_scalar = torch.cat((pre_fix_img_scalar, pre_grip_img_scalar, cur_fix_img_scalar, cur_grip_img_scalar),
                               axis=1)
    else:
        img_scalar = torch.cat((pre_fix_img_scalar, cur_fix_img_scalar), axis=1)
    if not use_s_loadar:
        if model.normalize:
            xref_pos_quat = torch.tensor(state_list).to(model.device)
            xref_pos_quat = model.normalize_pos_quat(xref_pos_quat).reshape((1, 14))
        else:
            xref_pos_quat = torch.tensor(state_list).reshape((1, 14)).to(model.device)
        xref_gripper = torch.tensor(grip_state_list).unsqueeze(0).to(model.device)
    else:
        pre_state = state_list[-2]
        cur_state = state_list[-1]
        if model.normalize:
            xref_pos_quat = torch.cat((pre_state, cur_state)).to(model.device)
            xref_pos_quat = model.normalize_pos_quat(xref_pos_quat).reshape((1, 14))
        else:
            xref_pos_quat = torch.cat((pre_state, cur_state)).reshape((1, 14)).to(model.device)
        pre_grip = grip_state_list[-2]
        cur_grip = grip_state_list[-1]
        xref_gripper = torch.cat((pre_grip, cur_grip)).reshape((1, 2)).to(model.device)
    return torch.cat((img_scalar, xref_pos_quat, xref_gripper), axis=1).float()


def infer(model, rpc, cfg, execute_steps=8, obs_horizon=2, ctr_grip=False, no_crop=True,
          use_image=False, crop_percent=0.2, guided_flow=False, save_folder='', ode_num=11):
    '''  experiment with RFMP on task Rotate Mug
        model: RFMP model
        rpc: interface with robot
        cfg: config file
        execute_steps: execution horizon
        obs_horizon: observation horizon
        ctr_grip: whether control gripper
        np_crop: True, no crop of image; False, Centercrop
        crop_percent: crop percent when no_crop False
        ode_num: ODE solving steps
        save_folder: folder to save experiment data
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
    fix_img, gripper_image = get_image(rs_recorder=rs_recorder, transform_cam=transform_cam, device=model.device,
                                       show_on_screen=True)

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
    total_infer_times = 0
    while True:
        print('********** total infer ' + str(total_infer_times) + '***************')
        total_infer_times += 1
        xref = get_ref(state_deque, grip_state_deque, fix_img_deque, grip_img_deque, model, use_image=use_image)
        # get actions series
        start_time = time.time()
        actions = model.sample_all(n_samples=1, device=model.device, xref=xref, different_sample=True,
                                   guided_flow=guided_flow, ode_steps=ode_num)
        infer_time_list.append(time.time() - start_time)
        print('inference uses ' + str(time.time() - start_time))
        actions = actions[-1].reshape((cfg.n_pred, model.dim))
        if cfg.normalize_pos_quat:
            actions[..., :-1] = model.denormalize_pos_quat(actions[..., :-1])
            actions[..., -1] = model.denormalize_grip(actions[..., -1])
        actions = actions.cpu().numpy()
        # print('predict actions ', actions)
        # print('state deque', state_deque)
        # print('pred action', actions[..., :3])
        for id, action in enumerate(actions[2:2 + execute_steps]):
            # print('execute action ', action)
            predict_action_list.append(action)
            # rpc.goto_cartesian_pose_nonblocking(action[:3], action[3:7], GENERATE_TRAJECTORY)
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
            # print('after execution, current pos is ', pose_loc)
            real_state_list.append(pose_loc)
            # print('cur state', pose_loc)
            state_deque.append(pose_loc)
            grip_state_deque.append(gripper_state)

            fix_img, gripper_image = get_image(rs_recorder=rs_recorder, transform_cam=transform_cam,
                                               device=model.device, fix_img_list=fix_img_list,
                                               use_grip_img=cfg.use_gripper_img, grip_img_list=grip_img_list,
                                               show_on_screen=True)
            fix_img_deque.append(fix_img)
            grip_img_deque.append(gripper_image)
            # fix_img_list.append(fix_img.cpu().numpy())
            step += 1
            # save_array_safe(fix_img_list, save_folder + '/img' + video_name + '.npy')
            # save_array_safe(real_state_list, save_folder + '/state' + video_name + '.npy')
            # save_array_safe(predict_action_list, save_folder + '/action' + video_name + '.npy')
            # save_array_safe(infer_time_list, save_folder + '/time' + video_name + '.npy')
            save_array_safe(infer_time_list, save_folder + '/time_' + str(ode_num - 1) + '.npy')
            # np.save(checkpoints_dir + '/' + video_name, fix_img_list)
            # np.save(checkpoints_dir + '/state' + video_name, real_state_list)
        # cv2.imshow("Press q to exit", np.zeros((1, 1)))
        # key = cv2.waitKey(1)
        # if key == ord('q'):
        #     break
    # np.save(checkpoints_dir + '/' + video_name, fix_img_list)
    # np.save(checkpoints_dir + '/state' + video_name, real_state_list)
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
    '''experiment with RFMP on real robot but get observation from dataloadar

        model: RFMP model
        rpc: interface with robot
        cfg: config file
        execute_steps: execution horizon
        obs_horizon: observation horizon
        data_loadar: data loadar
        pos_use_loadar: whether to use state information from dataloadar or read from real robot
        '''
    transform_cam = T.Compose([T.Resize((480, 640), antialias=None), ])
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
        # rpc.goto_cartesian_pose_nonblocking(batch['future_pos_quat'][:, 0, :3][0].cpu().numpy(), batch['future_pos_quat'][:, 0, 3:][0].cpu().numpy(), GENERATE_TRAJECTORY)
        # break
        pose_loc, gripper_state = get_info(rpc)
        fix_img = batch['observation']['v_fix']  # 3 * H * W
        plt.imshow(fix_img[0, -1].permute(1, 2, 0).numpy())
        plt.pause(0.2)
        plt.clf()
        gripper_img = batch['observation']['v_gripper']
        if i == 0:
            if not pos_use_loadar:
                state_deque = collections.deque([pose_loc] * obs_horizon, maxlen=obs_horizon)
                grip_state_deque = collections.deque([gripper_state] * obs_horizon, maxlen=obs_horizon)
            else:
                state_deque = collections.deque([batch['future_pos_quat'][:, 0, :]] * obs_horizon, maxlen=obs_horizon)
                grip_state_deque = collections.deque([batch['smooth_future_gripper'][:, 0, :]] * obs_horizon,
                                                     maxlen=obs_horizon)
            fix_img_deque = collections.deque([fix_img] * obs_horizon, maxlen=obs_horizon)
            grip_img_deque = collections.deque([gripper_img] * obs_horizon, maxlen=obs_horizon)
        else:
            if not pos_use_loadar:
                state_deque.append(pose_loc)
                grip_state_deque.append(gripper_state)
            else:
                state_deque.append(batch['future_pos_quat'][:, 0, :])
                grip_state_deque.append(batch['smooth_future_gripper'][:, 0, :])
            fix_img_deque.append(fix_img)
            grip_img_deque.append(gripper_img)
        real_state.append(pose_loc)
        demo_state.append(batch['future_pos_quat'][:, 0, :].cpu().numpy())

        # record the video of the real wordl experiment
        fix_img_real, gripper_image_real = get_image(rs_recorder=rs_recorder, transform_cam=transform_cam,
                                                     device=model.device, show_on_screen=False)
        fix_img_real_list.append(fix_img_real.cpu().numpy())

        if i % execute_steps == 0:
            xref = get_ref_infer_with_dataloader(state_deque, grip_state_deque, fix_img_deque, grip_img_deque, model,
                                                 use_s_loadar=pos_use_loadar)
            start_time = time.time()
            actions = model.sample_all(n_samples=1, device=model.device, xref=xref.to(model.device),
                                       different_sample=True)
            print('inference uses ' + str(time.time() - start_time))
            actions = actions[-1].reshape((cfg.n_pred, model.dim))
            if cfg.normalize_pos_quat:
                actions[..., :-1] = model.denormalize_pos_quat(actions[..., :-1])
            actions = actions.cpu().numpy()
            print('real state', pose_loc)
            print('state deque', state_deque)
            print('pred state', actions)
            pred_state.append(actions)
        cur_exc_step = int(i % execute_steps)
        rpc.goto_cartesian_pose_nonblocking(actions[cur_exc_step + 1][:3], actions[cur_exc_step + 1][3:7],
                                            GENERATE_TRAJECTORY)
        # time.sleep(0.5)
        np.save(checkpoints_dir + '/img_list_img_s_from_val_loadar1.npy', fix_img_real_list)
        np.save(checkpoints_dir + '/state_real.npy', real_state)
        np.save(checkpoints_dir + '/state_demo.npy', demo_state)
        np.save(checkpoints_dir + '/state_pred.npy', pred_state)
    # vwrite('infer.mp4', fix_img_list)


if __name__ == '__main__':
    add_info = '_vision_target_crop02_st250_old'  # _vision_new   _vision_new_smallercrop _vision_new_target_highfreq
    # _vision_new_target_crop03_newsimpledata_fixinitialpos_newnormrange
    # _vision_new_target_highfreq_newsimpledata_fixinitialpos_newnormrange
    cfg = OmegaConf.load('refcond_rfm_rotate_mug.yaml')
    cfg.model_type = 'Unet'
    checkpoints_dir = "checkpoints/checkpoints_rfm_" + cfg.data + \
                      "_n" + str(cfg.n_pred) + "_r" + str(cfg.n_ref) + "_c" + str(cfg.n_cond) + "_w" + str(
        cfg.w_cond) + cfg.model_type + add_info
    # model = ManifoldStateTrajectoriesMugRotateFMLitModule(cfg)
    model = ManifoldVisionTrajectoriesMugRotateFMLitModule(cfg)

    best_checkpoint = glob(checkpoints_dir + "/**/epoch**.ckpt", recursive=True)[0]
    last_checkpoint = './' + checkpoints_dir + '/last.ckpt'
    model_checkpoint = './' + checkpoints_dir + '/model.ckpt'
    model = model.load_from_checkpoint(best_checkpoint, cfg=cfg)
    model.to(torch.device('cuda'))
    model.small_var = True
    print('normalize', model.normalize)

    rpc = RPCInterface("10.87.170.254")  # 10.87.172.60

    data_folder_path = cfg.data_dir

    for _ in range(5):
        rpc.open_gripper()

    success = rpc.goto_home_joint()
    # target_pose_1 = np.array([0.47, 0.05, 0.24, 0.0, 1.0, 0.0, 0.0])
    target_pose_1 = np.array([0.38, 0.10, 0.36, 0.0, -0.95, 0.0, 0.0])
    # target_pose_1 = np.array([0.45356157237636696,
    #         0.054379275342792716,
    #         0.3723304611497262,
    #         0.15547613470374513,
    #         -0.986924090221739,
    #         0.03659647044615484,
    #         -0.021649711971741765])

    # best for simple dataset
    target_pose_1 = np.array([0.37888516710211556,
                              0.00510292887852903276,
                              0.35864418892044,
                              0.0030079931156787435,
                              -0.9972750881656733,
                              0.053389844167177965,
                              0.05082199367762835])

    # target_pose_1 = np.array([
    #         0.38220130741491637,
    #         0.010190241775998542,
    #         0.36850870711719125,
    #         0.01236478116009615,
    #         -0.9978602092071823,
    #         0.049083418598210055,
    #         0.04138759580567998
    #     ])
    # target_pose_1[0] += 0.02
    target_pose_1[1] += 0.05

    success = rpc.activate_cartesian_nonblocking_controller()

    rpc.goto_cartesian_pose_blocking(target_pose_1[:3], target_pose_1[3:], True)
    time.sleep(0.1)
    if success:
        print("controller activated successfully")
    else:
        print("control activation failed :(")
    execution_steps = 16
    ode_num = 2
    save_folder = checkpoints_dir + '/time'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    infer(model, rpc, cfg, steps=100, execute_steps=execution_steps, ctr_grip=True,
          video_name='2' + '_exc' + str(execution_steps), guided_flow=False,
          no_crop=False, use_image=True, crop_percent=0.2, ode_num=ode_num + 1, save_folder=save_folder)
    # infer_with_dataloader(model=model, rpc=rpc, cfg=cfg, data_loader=val_loader, pos_use_loadar=False)

    # reset the robot arm
    joint_config_1 = [0.0, -0.7, 0.035, -2.45, 0.0, 1.81, 0.73]
    success = rpc.goto_joint_position_blocking(joint_config_1, True)

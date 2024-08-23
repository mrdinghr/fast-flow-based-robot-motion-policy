import matplotlib.pyplot as plt

from stable_flow.stable_model_vision_rotatemug_pl_learntau import SRFMRotateMugVisionResnetTrajsModuleLearnTau

from data.real_robot.vision_audio_robot_arm import get_loaders
from omegaconf import OmegaConf
from glob import glob
from types import SimpleNamespace
import time
from tqdm import tqdm
import numpy as np
import torch
import cv2
import matplotlib

matplotlib.use('TkAgg')


'''
compare SRFMP generated result with recorded demonstration
'''


def compare_demo_gen(batch, action):
    demo_pos_quat = batch['future_pos_quat'][0]
    demo_grip_pos = batch['future_gripper'][0]
    gen_pos_quat = action[:, :7]
    gen_grip_pos = action[:, -1]
    plt.figure(figsize=(16, 16))
    for i in range(7):
        plt.subplot(8, 1, i + 1)
        plt.plot(demo_pos_quat[:, i], c='r', alpha=0.2, label='demo')
        plt.plot(gen_pos_quat[:, i], c='b', alpha=0.2, label='gen')
    plt.legend()
    plt.show()


def infer_whole_traj(data_loadar, model, execute_horizon=8, save_video=False, no_crop=True, ode_steps=5):
    gen_state_list = []
    demo_state_list = []
    demo_grip_list = []
    fix_img_list = []
    if save_video:
        fps = 3
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        width = 640
        height = 480
        video_writer = cv2.VideoWriter(checkpoints_dir + '/loadar_img.mp4', fourcc, fps, (width, height))
    for i, batch in tqdm(enumerate(data_loadar)):
        if i % execute_horizon == 0:
            _, xref, xcond, _ = model.unpack_predictions_reference_conditioning_samples(batch)
            start_time = time.time()
            actions = model.sample_all(1, model.device, xref=xref, xcond=xcond, different_sample=True,
                                       ode_steps=ode_steps, adp=False)
            print('tau series ', actions[:, 0, -1])
            actions = actions[-1, 0, :-1].reshape((cfg.n_pred, model.dim))

            if cfg.normalize_pos_quat:
                actions[..., :-1] = model.denormalize_pos_quat(actions[..., :-1])
                # actions[..., -1] = model.denormalize_grip(actions[..., -1])
            print('infer uses time ' + str(time.time() - start_time))
            if i == 0:
                gen_state_list = actions[:execute_horizon].cpu().numpy()
                regulate_quat_sign = torch.sign(
                    batch['traj']['target_pos_quat']['action'][0][:execute_horizon][..., 4]).unsqueeze(-1)
                demo_pos_quat = batch['traj']['target_pos_quat']['action'][0][:execute_horizon]
                demo_pos_quat[..., 3:] *= regulate_quat_sign
                demo_state_list = demo_pos_quat.cpu().numpy()
                demo_grip_list = batch['traj']['gripper']['action'][0][:execute_horizon].cpu().numpy()
            else:
                gen_state_list = np.concatenate([gen_state_list, actions[:execute_horizon].cpu().numpy()])
                regulate_quat_sign = torch.sign(
                    batch['traj']['target_pos_quat']['action'][0][:execute_horizon][..., 4]).unsqueeze(-1)
                demo_pos_quat = batch['traj']['target_pos_quat']['action'][0][:execute_horizon]
                demo_pos_quat[..., 3:] *= regulate_quat_sign
                demo_state_list = np.concatenate([demo_state_list, demo_pos_quat.cpu().numpy()])
                demo_grip_list = np.concatenate(
                    [demo_grip_list, batch['traj']['gripper']['action'][0][:execute_horizon].cpu().numpy()])
        obs = batch["observation"]
        image_f = obs["v_fix"][0][-1].permute(1, 2, 0).numpy()
        image_g = obs["v_gripper"][0][-1].permute(1, 2, 0).numpy()
        if save_video:
            video_writer.write(cv2.cvtColor((255 * image_f).astype(np.uint8), cv2.COLOR_RGB2BGR))
        fix_img_list.append((255 * image_f).astype(np.int_))
        image = np.concatenate([image_f, image_g], axis=0)
        plt.imshow(image)
        plt.draw()
        plt.pause(0.2)
        plt.clf()
    if save_video:
        video_writer.release()
    plt.close()
    # compare_demo_gen(batch, actions)
    plt.figure(figsize=(16, 16))
    for i in range(7):
        # plot the position and quaternion
        plt.subplot(8, 1, i + 1)
        plt.plot(demo_state_list[:, i], c='r', alpha=0.2, label='demo')
        plt.plot(gen_state_list[:, i], c='b', alpha=0.2, label='gen')
        if i > 2:
            plt.ylim([-1.1, 1.1])
        else:
            plt.ylim([0., 0.6])

        plt.legend()
    plt.subplot(8, 1, 8)
    plt.plot(gen_state_list[:, 7], c='b', alpha=0.2, label='gen')
    plt.plot(demo_grip_list, c='r', alpha=0.2, label='demo')
    plt.show()


if __name__ == '__main__':
    cfg = OmegaConf.load('srfm_rotate_mug.yaml')
    cfg.model_type = 'Unet'
    data_folder = cfg.data_dir
    data_args = SimpleNamespace()
    data_args.ablation = 'vf_vg'
    data_args.num_stack = 2
    data_args.frameskip = 1
    data_args.no_crop = False
    data_args.crop_percent = 0.2
    data_args.resized_height_v = cfg.image_height
    data_args.resized_width_v = cfg.image_width
    data_args.len_lb = cfg.n_pred - 1
    data_args.sampling_time = 250
    data_args.source = False
    data_args.catg = "resample"
    data_args.norm_type = "limit"
    data_args.smooth_factor = (1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5)
    add_info = 'lambdataux25_harddata'
    train_loader, val_loader, _ = get_loaders(batch_size=cfg.optim.batch_size, args=data_args, data_folder=data_folder,
                                              drop_last=False, debug=True)
    checkpoints_dir = "checkpoints/checkpoints_rfm_" + cfg.data + \
                      "_n" + str(cfg.n_pred) + "_r" + str(cfg.n_ref) + "_c" + str(cfg.n_cond) + "_w" + str(
        cfg.w_cond) + cfg.model_type + add_info
    model = SRFMRotateMugVisionResnetTrajsModuleLearnTau(cfg)

    best_checkpoint = glob(checkpoints_dir + "/**/epoch**.ckpt", recursive=True)[0]
    last_checkpoint = './' + checkpoints_dir + '/last.ckpt'
    model_checkpoint = './' + checkpoints_dir + '/model.ckpt'
    model = model.load_from_checkpoint(best_checkpoint, cfg=cfg)
    print(model)
    model.small_var = True
    model.to(torch.device('cuda'))
    infer_whole_traj(val_loader, model, execute_horizon=8, save_video=False, ode_steps=1)

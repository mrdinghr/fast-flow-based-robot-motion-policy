import numpy as np
import os


def cal_smoothnes(pos_series):
    vel = np.diff(pos_series, axis=0)
    acc = np.diff(vel, axis=0)
    jerk = np.diff(acc, axis=0)
    jerk = np.sum(np.linalg.norm(jerk, axis=1)) / len(jerk)
    return jerk


if __name__ == '__main__':
    folder = '/home/dia1rng/hackathon/flow-matching-policies/examples/rotate_mug/checkpoints/checkpoints_rfm_rotate_mug_n16_r2_c0_w0Unet_vision_target_crop02_st250_new'
    files = os.listdir(folder)
    file_name = 'actionfix_img_list_test3_exc8'
    action_series = np.load(folder + '/' + file_name + '.npy')
    jerk = cal_smoothnes(action_series)


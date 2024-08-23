from omegaconf import OmegaConf
from manifm.datasets import _get_dataset
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from plot_utils.plot_sphere_matplotlib import plot_sphere
import time
from PIL import Image
# from plot_utils.plot_sphere_mayavi import plot_sphere as plot_sphere_mayavi
# from mayavi import mlab


'''
create dataset for sphere pusht task

Please note: we don't really build a Sphere PushT env in Mujoco, we still use PushT env from Diffusion Policy
we project the observation and task space to a Sphere
'''


def plt3d_sphere_project_img(img, save_dir='', scale=1):
    '''
    project the euclidean pusht iamge to sphere pusht image

    img: image of euclidean PushT env
    save_dir: the folder to save the image after projected on sphere
    scale: the ratio of lenght of origin Euclidean PushT env and Sphere diameter

    return: image of sphere PushT env
    '''
    # set radius of sphere
    r = 48 / scale

    # start_time = time.time()
    fig = plt.figure(constrained_layout=True, figsize=(1, 1))
    ax = fig.add_subplot(projection='3d')
    x_range = np.arange(-48, 48)
    y_range = np.arange(-48, 48)
    X, Y = np.meshgrid(x_range, y_range)
    K = ((X ** 2 + Y ** 2 + r ** 2) / r ** 2) ** 0.5
    SP_X  = X / K
    SP_Y = Y / K
    SP_Z = r / K
    ax.scatter3D(SP_X, SP_Y, SP_Z, c=img.reshape(96 * 96, 3) / 255, s=1)
    plot_sphere(ax, r=r, alpha=0.4, lim=r * 1.1)
    ax.set_proj_type('ortho')
    ax.axis('off')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    # ax._axis3don = False
    ax.view_init(90, -90)
    # plot_finish_time = time.time()
    # print('plot use ' + str(plot_finish_time - start_time))
    # plt.show()
    plt.savefig(save_dir + '.png', bbox_inches='tight', dpi=110, pad_inches=0)
    # plt.show()
    plt.close()
    sp_img = Image.open(save_dir + '.png')
    sp_rgb_image = sp_img.convert('RGB')
    sp_rgb_data = np.array(sp_rgb_image)
    # plt.imshow(sp_img)
    # print('save use ' + str(time.time() - plot_finish_time))
    # np.save('rgb.npy', rgb)
    return sp_rgb_data


# project agent position to sphere
# agent pos is in range(0, 512)
def project_to_sphere_agent_pos(agent_pos, r=256):
    '''
    project agent position of euclidean pusht to sphere

    agent_pos: origin agent position on Euclidean PushT env
    r: radius of sphere to project

    return: agent_pos on Sphere PushT env
    '''
    data = np.hstack((agent_pos, r * np.ones((agent_pos.shape[0], 1))))
    data = data * r / np.linalg.norm(data, axis=1)[:, None]
    # data += center
    return data


# project action on sphere back to tangenet space
# sphere with radius 512
def project_sphere_action_back(sp_action, r=256):
    '''
    project the sphere pusht agent position back to euclidean space

    sp_action: action on Sphere PushT env
    r: sphere radius

    return: action on Euclidean PushT
    '''
    k = r / sp_action[:, -1]
    data = sp_action[:, :2] * k.reshape((len(k), 1))
    return data


# create dataset of image
def create_img_sphere_dataset(scale=1.5):
    '''
    create the sphere pusht image dataset
    '''

    cfg = OmegaConf.load('./../riemannianfm_pusht/refcond_rfm_euclidean_vision_pusht.yaml')
    dataset, _ = _get_dataset(cfg)
    images = dataset.replay_buffer.data['img']
    sphere_img_dataset = []
    for id, image_data in tqdm(enumerate(images)):
        sp_data = plt3d_sphere_project_img(image_data, save_dir='./sp_img_test/' + str(id), scale=scale)
        sphere_img_dataset.append(sp_data)
        # plt.figure()
        # fig = plt.imshow(image_data.astype(np.int_))
        # plt.xticks([])
        # plt.yticks([])
        # plt.pause(0.3)
        # plt.close()
        # plt.show()
        np.save('sphere_img_test.npy', sphere_img_dataset)
    # sphere_img_dataset = np.array(sphere_img_dataset).moveaxis()
    # np.save('sphere_img_3d_dataset_less_blank_scale15.npy', sphere_img_dataset)


# create dataset of agent position and action
def create_norm_agent_pos_sphere_dataset(scale=2):
    '''
    create sphere pusht agent position and action dataset
    '''
    cfg = OmegaConf.load('./../riemannianfm_pusht/refcond_rfm_euclidean_vision_pusht.yaml')
    dataset, _ = _get_dataset(cfg)
    agent_pos_list = dataset.replay_buffer.data['state'][:, :2]
    action_list = dataset.replay_buffer.data['action']
    norm_agent_pos_list = (agent_pos_list - 256) / 256
    norm_action_list = (action_list - 256) / 256
    # the data to [-1, 1]
    sphere_agent_pos_list = project_to_sphere_agent_pos(norm_agent_pos_list * scale, r=1)
    sphere_action_list = project_to_sphere_agent_pos(norm_action_list * scale, r=1)
    np.save('sphere_agent_pos_normfirst' + str(scale) + '.npy', sphere_agent_pos_list)
    np.save('sphere_action_normfirst' + str(scale) + '.npy', sphere_action_list)


if __name__ == '__main__':
    # create_norm_agent_pos_sphere_dataset(scale=1.5)
    create_img_sphere_dataset()

    # create_agent_pos_sphere_dataset()
    '''check whether the project function written correct'''
    # cfg = OmegaConf.load('./../riemannianfm_pusht/refcond_rfm_euclidean_vision_pusht.yaml')
    # dataset, _ = _get_dataset(cfg)
    # sphere_action = np.load('sphere_action.npy')
    # sphere_agent_pos = np.load('sphere_agent_pos.npy')
    # action = project_sphere_action_back(sphere_action)
    # agent_pos = project_sphere_action_back(sphere_agent_pos)

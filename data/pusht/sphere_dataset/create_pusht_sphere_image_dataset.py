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
the image is project to sphere with radius half width of picture

the action and agent pos is project to sphere with r 256
'''


def sphere_project_img(img):
    rows = img.shape[0]
    cols = img.shape[1]
    blank = np.zeros_like(img)
    # blank = np.zeros((4000,4000,3))
    # set the center of plot
    center_x = int(rows / 2)
    center_y = int(cols / 2)
    # set radius of sphere
    r = int(((rows ** 2 + cols ** 2) ** 0.5))
    # r = int(rows / 2)
    # r = 1
    # assume z plane at z = r
    for org_x in range(rows):
        for org_y in range(cols):
            org_relative_x = org_x - center_x
            org_relative_y = org_y - center_y
            k = ((org_relative_x ** 2 + org_relative_y ** 2 + r ** 2) / r ** 2) ** 0.5
            sp_x = int(org_relative_x / k) + center_x
            sp_y = int(org_relative_y / k) + center_y
            blank[sp_x, sp_y, :] = img[org_x, org_y, :]
    # plt.imshow(blank.astype(np.int_))
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()
    # cv2.imwrite('out.jpg', blank)
    return blank


def right_sphere_project_img(img):
    rows = img.shape[0]
    cols = img.shape[1]
    blank = np.zeros_like(img)
    # blank = np.zeros((4000,4000,3))
    # set the center of plot
    center_x = int(rows / 2)
    center_y = int(cols / 2)
    # set radius of sphere
    # r = int(((rows ** 2 + cols ** 2) ** 0.5))
    r = int(rows / 2)
    # r = 1
    # assume z plane at z = r
    fig, ax = plt.subplots(figsize=(8, 8), dpi=96)
    for org_x in range(rows):
        for org_y in range(cols):
            org_relative_x = org_x - center_x
            org_relative_y = org_y - center_y
            k = ((org_relative_x ** 2 + org_relative_y ** 2 + r ** 2) / r ** 2) ** 0.5
            sp_x = int(org_relative_x / k) + center_x
            sp_y = int(org_relative_y / k) + center_y
            blank[sp_x, sp_y, :] = img[org_x, org_y, :]
    plt.imshow(blank.astype(np.int_))
    plot_sphere(ax)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    # cv2.imwrite('out.jpg', blank)
    return blank


# def plt3d_sphere_project_img(img, save_dir=''):
#     rows = img.shape[0]
#     cols = img.shape[1]
#     center_x = int(rows / 2)
#     center_y = int(cols / 2)
#     # set radius of sphere
#     r = int(((center_x ** 2 + center_y ** 2) ** 0.5))
#
#     start_time = time.time()
#     fig = plt.figure(constrained_layout=True)
#     ax = fig.add_subplot(projection='3d')
#     for org_x in tqdm(range(rows)):
#         for org_y in range(cols):
#             org_relative_x = org_x - center_x
#             org_relative_y = org_y - center_y
#             k = ((org_relative_x ** 2 + org_relative_y ** 2 + r ** 2) / r ** 2) ** 0.5
#             sp_x = org_relative_x / k
#             sp_y = org_relative_y / k
#             sp_z = r / k
#             ax.scatter(sp_x, sp_y, sp_z, color=img[org_x, org_y] / 255)
#     plot_sphere(ax, r=r, alpha=0.4, lim=r * 1.1)
#     ax.set_proj_type('ortho')
#     # plt.imshow(blank.astype(np.int_))
#     # plt.xlim([-48, 48])
#     # plt.ylim([-48, 48])
#     ax.axis('off')
#     ax.set_xlabel('')
#     ax.set_ylabel('')
#     ax.set_zlabel('')
#     ax._axis3don = False
#     ax.view_init(90, -90)
#     plot_finish_time = time.time()
#     print('plot use ' + str(plot_finish_time - start_time))
#     plt.savefig(save_dir + '.png', bbox_inches='tight')
#     print('save use ' + str(time.time() - plot_finish_time))
#     # np.save('rgb.npy', rgb)
#     plt.show()
#     return


def plt3d_sphere_project_img(img, save_dir='', scale=1):
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


# run too too slow, doesn't work
def mayavi_sphere_project_img(img):
    rows = img.shape[0]
    cols = img.shape[1]
    center_x = int(rows / 2)
    center_y = int(cols / 2)
    # set radius of sphere
    r = int(((center_x ** 2 + center_y ** 2) ** 0.5))

    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(700, 700))
    fig = mlab.gcf()
    for org_x in tqdm(range(rows)):
        for org_y in range(cols):
            org_relative_x = org_x - center_x
            org_relative_y = org_y - center_y
            k = ((org_relative_x ** 2 + org_relative_y ** 2 + r ** 2) / r ** 2) ** 0.5
            sp_x = org_relative_x / k
            sp_y = org_relative_y / k
            sp_z = r / k
            mlab.points3d(sp_x, sp_y, sp_z, color=tuple(img[org_x, org_y] / 255))
    plot_sphere_mayavi(fig, radius=r)
    mlab.show()
    # np.save('rgb.npy', rgb)
    return


# agent pos is in range(0, 512)
def project_to_sphere_agent_pos(agent_pos, r=256):
    data = np.hstack((agent_pos, r * np.ones((agent_pos.shape[0], 1))))
    data = data * r / np.linalg.norm(data, axis=1)[:, None]
    # data += center
    return data


# sphere with radius 512
def project_sphere_action_back(sp_action, r=256):
    k = r / sp_action[:, -1]
    data = sp_action[:, :2] * k.reshape((len(k), 1))
    return data


def create_img_sphere_dataset(scale=1.5):
    cfg = OmegaConf.load('./../riemannianfm_pusht/refcond_rfm_euclidean_vision_pusht.yaml')
    dataset, _ = _get_dataset(cfg)
    images = dataset.replay_buffer.data['img']
    sphere_img_dataset = []
    for id, image_data in tqdm(enumerate(images)):
        # sp_data = new_sphere_project_img(image_data)
        # plt.imshow(image_data.astype(np.int_))
        # plt.show()
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



# this function is wrong
# this function first project data to sphere with radius 256, and then normalization, so the result is very weir,
# we dont know the final data is on sphere of which form
def create_agent_pos_sphere_dataset():
    cfg = OmegaConf.load('./../riemannianfm_pusht/refcond_rfm_euclidean_vision_pusht.yaml')
    dataset, _ = _get_dataset(cfg)
    agent_pos_list = dataset.replay_buffer.data['state'][:, :2]
    action_list = dataset.replay_buffer.data['action']
    sp_agent_pos = project_to_sphere_agent_pos(agent_pos_list)
    sp_action = project_to_sphere_agent_pos(action_list)
    np.save('sphere_agent_pos.npy', sp_agent_pos)
    np.save('sphere_action.npy', sp_action)
    return sp_agent_pos, sp_action


def create_norm_agent_pos_sphere_dataset(scale=2):
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

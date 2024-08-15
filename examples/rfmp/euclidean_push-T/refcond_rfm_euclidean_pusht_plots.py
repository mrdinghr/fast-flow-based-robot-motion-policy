import os
import numpy as np
import torch
from omegaconf import OmegaConf
from glob import glob
import dill
import wandb
import json
import pathlib
from tqdm.auto import tqdm
from IPython.display import Video
from skvideo.io import vwrite
import collections

import matplotlib
matplotlib.use('TkAgg')

from manifm.datasets import _get_dataset
from manifm.model_trajectories_pl import ManifoldTrajectoriesFMLitModule
from diffusion_policy.env.pusht.pusht_env import PushTEnv
from diffusion_policy.dataset.pusht_state_dataset import normalize_data, unnormalize_data

# if torch.cuda.is_available():
#     device = torch.cuda.current_device()
# else:
#     device = 'cpu'
device = 'cpu'


def sample_dataset(dataset, n):
    idx = np.random.randint(dataset.shape[0], size=n)
    return dataset[idx]


def trajectory_generation(model, xref, xcond, n_steps, n_actions=1):
    trajectory = []
    nb_traj = xref.shape[0]
    dim = model.dim
    n_ref = model.n_ref
    n_cond = model.n_cond
    w_cond = model.w_cond

    if n_cond == 0:
        xcond = torch.zeros((nb_traj, 0))

    for n in range(n_steps):
        ode_traj = model.sample_all(nb_traj, xref.device, xref, xcond)
        actions = ode_traj[-1][None][..., :dim * n_actions].squeeze().reshape(nb_traj, n_actions, dim)
        if n == 0:
            trajectory = torch.hstack((xref.reshape(nb_traj, n_ref, dim), actions))
        else:
            trajectory = torch.hstack((trajectory, actions))

        xref = trajectory[:, -n_ref:, :].reshape(nb_traj, -1)
        if n_cond > 0:
            len_traj = trajectory.shape[1]
            if w_cond > 0:
                start_w_cond = np.maximum(0, len_traj - n_ref - w_cond)
            else:
                start_w_cond = 0
            idx_cond = np.random.randint(start_w_cond, high=len_traj - n_ref, size=nb_traj)
            xcond = torch.hstack(
                (trajectory[np.arange(0, nb_traj), idx_cond, :dim], torch.Tensor(len_traj - idx_cond)[:, None]))

    return trajectory


# def evaluate(workspace, checkpoint, output_dir):
#     if os.path.exists(output_dir):
#         click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
#     pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
#
#     # load checkpoint
#     payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
#     cfg = payload['cfg']
#
#     workspace.load_payload(payload, exclude_keys=None, include_keys=None)
#
#     # get policy from workspace
#     policy = workspace.model
#     if cfg.training.use_ema:
#         policy = workspace.ema_model
#
#     policy.to(device)
#     policy.eval()
#
#     # run eval
#     env_runner = workspace.env_runner
#     runner_log = env_runner.run(policy)
#
#     # dump log to json
#     json_log = dict()
#     for key, value in runner_log.items():
#         if isinstance(value, wandb.sdk.data_types.video.Video):
#             json_log[key] = value._path
#         else:
#             json_log[key] = value
#     out_path = os.path.join(output_dir, 'eval_log.json')
#     json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)


def inference(stats, model, obs_horizon, pred_horizon, action_horizon):
    # limit enviornment interaction to 200 steps before termination
    max_steps = 500
    env = PushTEnv()
    # use a seed >200 to avoid initial states seen in the training dataset
    env.seed(6666)
    # env.seed(100001)

    # get first observation
    # obs, info = env.reset()
    obs = env.reset()

    # keep a queue of last 2 steps of observations
    obs_deque = collections.deque(
        [obs] * obs_horizon, maxlen=obs_horizon)
    # save visualization and rewards
    imgs = [env.render(mode='rgb_array')]
    rewards = list()
    done = False
    step_idx = 0

    with tqdm(total=max_steps, desc="Eval PushTStateEnv") as pbar:
        while not done:
            B = 1
            # stack the last obs_horizon (2) number of observations
            obs_seq = np.stack(obs_deque)
            # normalize observation
            nobs = normalize_data(obs_seq, stats=stats['obs'])
            # device transfer
            nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)

            # infer action
            # reshape observation to (B,obs_horizon*obs_dim)
            obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

            # Flow matching action
            ode_traj = model.sample_all(B, obs_cond.device, obs_cond,
                                        torch.zeros(B, 0, dtype=obs_cond.dtype, device=obs_cond.device))
            naction = ode_traj[-1][None][..., :model.dim * pred_horizon].squeeze().reshape(B, pred_horizon, model.dim)

            # unnormalize action
            naction = naction.detach().to('cpu').numpy()
            # (B, pred_horizon, action_dim)
            naction = naction[0]
            action_pred = unnormalize_data(naction, stats=stats['action'])

            # only take action_horizon number of actions
            start = obs_horizon - 1
            end = start + action_horizon
            action = action_pred[start:end, :]
            # (action_horizon, action_dim)

            # execute action_horizon number of steps
            # without replanning
            for i in range(len(action)):
                # stepping env
                obs, reward, done, info = env.step(action[i])
                # save observations
                obs_deque.append(obs)
                # and reward/vis
                rewards.append(reward)
                imgs.append(env.render(mode='rgb_array'))

                # update progress bar
                step_idx += 1
                pbar.update(1)
                pbar.set_postfix(reward=reward)
                if step_idx > max_steps:
                    done = True
                if done:
                    break

    # print out the maximum target coverage
    print('Score: ', max(rewards))

    # visualize
    vwrite(checkpoints_dir + '/' + cfg.model_type + '.mp4', imgs)
    Video('vis.mp4', embed=True, width=256, height=256)


if __name__ == '__main__':
    # Load config
    cfg = OmegaConf.load('refcond_rfm_euclidean_pusht.yaml')
    # checkpoints_dir = "checkpoints/checkpoints_rfm_" + cfg.data + "_" + cfg.letter[0] + "_n" + str(cfg.n_pred)
    checkpoints_dir = "checkpoints/checkpoints_rfm_" + cfg.data + \
                      "_n" + str(cfg.n_pred) + "_r" + str(cfg.n_ref) + "_c" + str(cfg.n_cond) + "_w" + str(
        cfg.w_cond) + cfg.model_type

    # Number of steps and actions per step
    n_actions = 8
    n_steps = int(200 / n_actions)

    # Load dataset
    dataset, _ = _get_dataset(cfg)

    # Construct model
    model = ManifoldTrajectoriesFMLitModule(cfg)
    print(model)

    # best_checkpoint = glob(checkpoints_dir + "/**/epoch**.ckpt", recursive=True)[0]
    last_checkpoint = './' + checkpoints_dir + '/last.ckpt'
    model_checkpoint = './' + checkpoints_dir + '/model.ckpt'
    model = model.load_from_checkpoint(last_checkpoint, cfg=cfg)

    # Training data statistics (min, max) for each dim
    stats = dataset.stats

    # Inference
    inference(stats, model, obs_horizon=cfg.n_ref, pred_horizon=cfg.n_pred, action_horizon=int(cfg.n_pred / 2))

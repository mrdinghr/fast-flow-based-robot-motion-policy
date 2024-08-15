"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import os
from pathlib import Path
from csv import reader
import random
import scipy.io as sp
import numpy as np
import pandas as pd
import igl
import torch
from torch.utils.data import Dataset, DataLoader

from manifm.manifolds import Euclidean, Sphere, FlatTorus, Mesh, SPD, PoincareBall, ProductManifold, \
    ProductManifoldTrajectories
from manifm.manifolds.mesh import Metric
from manifm.utils import cartesian_from_latlon

from diffusion_policy.dataset.pusht_dataset import PushTLowdimDataset
from diffusion_policy.dataset.pusht_state_dataset import PushTStateDataset
from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset
from diffusion_policy.common.pytorch_util import dict_apply
from typing import Dict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = Path(CURRENT_DIR).parent.resolve()


def load_csv(filename):
    file = open(filename, "r")
    lines = reader(file)
    dataset = np.array(list(lines)[1:]).astype(np.float64)
    return dataset


def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats


def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata


class EarthData(Dataset):
    manifold = Sphere()
    dim = 3

    def __init__(self, dirname, filename):
        filename = os.path.join(dirname, filename)
        dataset = load_csv(filename)
        dataset = torch.Tensor(dataset)
        self.latlon = dataset
        self.data = cartesian_from_latlon(dataset / 180 * np.pi)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Volcano(EarthData):
    def __init__(self, dirname):
        super().__init__(dirname, "volcano.csv")


class Earthquake(EarthData):
    def __init__(self, dirname):
        super().__init__(dirname, "earthquake.csv")


class Fire(EarthData):
    def __init__(self, dirname):
        super().__init__(dirname, "fire.csv")


class Flood(EarthData):
    def __init__(self, dirname):
        super().__init__(dirname, "flood.csv")


class Top500(Dataset):
    manifold = FlatTorus()
    dim = 2

    def __init__(self, root="data/top500", amino="General"):
        data = pd.read_csv(
            f"{root}/aggregated_angles.tsv",
            delimiter="\t",
            names=["source", "phi", "psi", "amino"],
        )

        amino_types = ["General", "Glycine", "Proline", "Pre-Pro"]
        assert amino in amino_types, f"amino type {amino} not implemented"

        data = data[data["amino"] == amino][["phi", "psi"]].values.astype("float32")
        self.data = torch.tensor(data % 360 * np.pi / 180)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class RNA(Dataset):
    manifold = FlatTorus()
    dim = 7

    def __init__(self, root="data/rna"):
        data = pd.read_csv(
            f"{root}/aggregated_angles.tsv",
            delimiter="\t",
            names=[
                "source",
                "base",
                "alpha",
                "beta",
                "gamma",
                "delta",
                "epsilon",
                "zeta",
                "chi",
            ],
        )

        data = data[
            ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi"]
        ].values.astype("float32")
        self.data = torch.tensor(data % 360 * np.pi / 180)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MeshDataset(Dataset):
    dim = 3

    def __init__(self, root: str, data_file: str, obj_file: str, scale=1 / 250):
        with open(os.path.join(root, data_file), "rb") as f:
            data = np.load(f)

        v, f = igl.read_triangle_mesh(os.path.join(root, obj_file))

        self.v = torch.tensor(v).float() * scale
        self.f = torch.tensor(f).long()
        self.data = torch.tensor(data).float() * scale

    def manifold(self, *args, **kwargs):
        return Mesh(self.v, self.f, *args, **kwargs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SimpleBunny(MeshDataset):
    def __init__(self, root="data/mesh"):
        super().__init__(
            root=root,
            data_file="bunny_simple.npy",
            obj_file="bunny_simp.obj",
            scale=1 / 250,
        )


class Bunny10(MeshDataset):
    def __init__(self, root="data/mesh"):
        super().__init__(
            root=root,
            data_file="bunny_eigfn009.npy",
            obj_file="bunny_simp.obj",
            scale=1 / 250,
        )


class Bunny50(MeshDataset):
    def __init__(self, root="data/mesh"):
        super().__init__(
            root=root,
            data_file="bunny_eigfn049.npy",
            obj_file="bunny_simp.obj",
            scale=1 / 250,
        )


class Bunny100(MeshDataset):
    def __init__(self, root="data/mesh"):
        super().__init__(
            root=root,
            data_file="bunny_eigfn099.npy",
            obj_file="bunny_simp.obj",
            scale=1 / 250,
        )


class Spot10(MeshDataset):
    def __init__(self, root="data/mesh"):
        super().__init__(
            root=root,
            data_file="spot_eigfn009.npy",
            obj_file="spot_simp.obj",
            scale=1.0,
        )


class Spot50(MeshDataset):
    def __init__(self, root="data/mesh"):
        super().__init__(
            root=root,
            data_file="spot_eigfn049.npy",
            obj_file="spot_simp.obj",
            scale=1.0,
        )


class Spot100(MeshDataset):
    def __init__(self, root="data/mesh"):
        super().__init__(
            root=root,
            data_file="spot_eigfn099.npy",
            obj_file="spot_simp.obj",
            scale=1.0,
        )


class HyperbolicDatasetPair(Dataset):
    manifold = PoincareBall()
    dim = 2

    def __init__(self, distance=0.6, std=0.7):
        self.distance = distance
        self.std = std

    def __len__(self):
        return 20000

    def __getitem__(self, idx):
        sign0 = (torch.rand(1) > 0.5).float() * 2 - 1
        sign1 = (torch.rand(1) > 0.5).float() * 2 - 1

        mean0 = torch.tensor([self.distance, self.distance]) * sign0
        mean1 = torch.tensor([-self.distance, self.distance]) * sign1

        x0 = PoincareBall().wrapped_normal(2, mean=mean0, std=self.std)
        x1 = PoincareBall().wrapped_normal(2, mean=mean1, std=self.std)

        return {"x0": x0, "x1": x1}


class MeshDatasetPair(Dataset):
    dim = 3

    def __init__(self, root: str, data_file: str, obj_file: str, scale: float):
        data = np.load(os.path.join(root, data_file), "rb")
        x0 = data["x0"]
        x1 = data["x1"]

        self.Z0 = float(data["Z0"])
        self.Z1 = float(data["Z1"])

        if "std" in data:
            self.std = float(data["std"])
        else:
            # previous default
            self.std = 1 / 9.5

        v, f = igl.read_triangle_mesh(os.path.join(root, obj_file))

        self.v = torch.tensor(v).float() * scale
        self.f = torch.tensor(f).long()

        self.x0 = torch.tensor(x0).float() * scale
        self.x1 = torch.tensor(x1).float() * scale

    def manifold(self, *args, **kwargs):
        def base_logprob(x):
            x = (x[..., :2] - 0.5) / self.std
            logZ = -0.5 * np.log(2 * np.pi)
            logprob = logZ - x.pow(2) / 2
            logprob = logprob - np.log(self.std)
            return logprob.sum(-1) - np.log(self.Z0)

        mesh = Mesh(self.v, self.f, *args, **kwargs)
        mesh.base_logprob = base_logprob
        return mesh

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, idx):
        idx0 = int(len(self.x0) * random.random())
        return {"x0": self.x0[idx0], "x1": self.x1[idx]}


class Maze3v2(MeshDatasetPair):
    def __init__(self, root="data/mesh"):
        super().__init__(
            root=root, data_file="maze_3x3v2.npz", obj_file="maze_3x3.obj", scale=1 / 3
        )


class Maze4v2(MeshDatasetPair):
    def __init__(self, root="data/mesh"):
        super().__init__(
            root=root, data_file="maze_4x4v2.npz", obj_file="maze_4x4.obj", scale=1 / 4
        )


class Wrapped(Dataset):
    def __init__(
        self,
        manifold,
        dim,
        n_mixtures=1,
        scale=0.2,
        centers=None,
        dataset_size=200000,
    ):
        self.manifold = manifold
        self.dim = dim
        self.n_mixtures = n_mixtures
        if centers is None:
            self.centers = self.manifold.random_uniform(n_mixtures, dim)
        else:
            self.centers = centers
        self.scale = scale
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        del idx

        idx = torch.randint(self.n_mixtures, (1,)).to(self.centers.device)
        mean = self.centers[idx].squeeze(0)

        tangent_vec = torch.randn(self.dim).to(self.centers)
        tangent_vec = self.manifold.proju(mean, tangent_vec)
        tangent_vec = self.scale * tangent_vec
        sample = self.manifold.expmap(mean, tangent_vec)
        return sample


class ExpandDataset(Dataset):
    def __init__(self, dset, expand_factor=1):
        self.dset = dset
        self.expand_factor = expand_factor

    def __len__(self):
        return len(self.dset) * self.expand_factor

    def __getitem__(self, idx):
        return self.dset[idx % len(self.dset)]


class EuclideanLasaDataset(Dataset):
    def __init__(self, dirname, letter, subsampling=5, normalize=True):
        self.manifold = Euclidean()
        self.dim = 2

        dataset_path = os.path.join(dirname, letter + '.mat')
        letter_data = sp.loadmat(dataset_path)['demos'][0]
        nb_demos = len(letter_data)
        demos = []
        for i in range(nb_demos):
            data = [letter_data[i][0][0][0].T]
            demos += data

        demos = np.array(demos)[:, ::subsampling, :]
        stacked_demos = np.vstack(demos)

        if normalize:
            max_demos = np.max(stacked_demos, 0)
            min_demos = np.min(stacked_demos, 0)

            demos = 2 * (demos - min_demos) / (max_demos - min_demos) - 1.0

        stacked_demos = np.vstack(demos)

        self.demos = torch.Tensor(demos)
        self.data = torch.Tensor(stacked_demos)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SphereLasaDataset(Dataset):
    def __init__(self, dirname, letter, subsampling=5):
        self.manifold = Sphere()
        self.dim = 3

        dataset_path = os.path.join(dirname, letter + '.mat')
        letter_data = sp.loadmat(dataset_path)['demos'][0]
        nb_demos = len(letter_data)
        demos = []
        for i in range(nb_demos):
            data = [letter_data[i][0][0][0].T]
            demos += data

        # Output on a sphere
        demos_on_sphere = []
        scale = 0.08
        for i in range(nb_demos):
            data = np.copy(demos[i])
            offset = - np.min(data, axis=0) / (np.max(data, axis=0) - np.min(data, axis=0)) - 0.5
            # data = data * scale + offset
            data = data * scale - np.mean(data * scale, axis=0)
            data = np.hstack((data, np.ones((data.shape[0], 1))))
            data = data / np.linalg.norm(data, axis=1)[:, None]

            demos_on_sphere += [data]

        demos_on_sphere = np.array(demos_on_sphere)[:, ::subsampling, :]
        stacked_demos_on_sphere = np.vstack(demos_on_sphere)
        self.demos = torch.Tensor(demos_on_sphere)
        self.data = torch.Tensor(stacked_demos_on_sphere)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class EuclideanMultimodalDataset(Dataset):
    def __init__(self, dirname, subsampling=5, normalize=True):
        self.manifold = Euclidean()
        self.dim = 2

        dataset_path = os.path.join(dirname, 'LShape.mat')
        letter_data = sp.loadmat(dataset_path)['demos'][0]
        nb_demos = len(letter_data)
        demos = []
        for i in range(nb_demos):
            data = [letter_data[i][0][0][0].T]
            demos += data

        demos = np.array(demos)[:, ::subsampling, :]
        stacked_demos = np.vstack(demos)

        max_demos = np.max(stacked_demos, 0)
        demos = demos - max_demos
        demos_mirror = demos * np.array([-1, 1])
        demos = np.vstack((demos, demos_mirror))
        stacked_demos = np.vstack(demos)

        if normalize:
            max_demos = np.max(stacked_demos, 0)
            min_demos = np.min(stacked_demos, 0)

            demos = 2 * (demos - min_demos) / (max_demos - min_demos) - 1.0

        demos = np.flip(demos, 1)
        stacked_demos = np.vstack(np.flip(demos, 1))
        self.demos = torch.Tensor(demos.copy())
        self.data = torch.Tensor(stacked_demos)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SphereMultimodalDataset(Dataset):
    def __init__(self, dirname, subsampling=5, normalize=True):
        self.manifold = Sphere()
        self.dim = 3

        dataset_path = os.path.join(dirname, 'LShape.mat')
        letter_data = sp.loadmat(dataset_path)['demos'][0]
        nb_demos = len(letter_data)
        demos = []
        for i in range(nb_demos):
            data = [letter_data[i][0][0][0].T]
            demos += data

        # demos = np.array(demos)[:, ::subsampling, :] / 2.
        # stacked_demos = np.vstack(demos)
        #
        # max_demos = np.max(stacked_demos, 0)
        # demos = demos - max_demos
        # demos_mirror = demos * np.array([-1, 1])
        # demos = np.vstack((demos, demos_mirror))
        # stacked_demos = np.vstack(demos)
        # #
        # # if normalize:
        # #     max_demos = np.max(stacked_demos, 0)
        # #     min_demos = np.min(stacked_demos, 0)
        # #
        # #     demos = 2 * (demos - min_demos) / (max_demos - min_demos) - 1.0
        #
        # demos = np.flip(demos, 1)
        # stacked_demos = np.vstack(np.flip(demos, 1))

        # Output on a sphere
        demos_on_sphere = []
        scale = 0.04
        for i in range(nb_demos):
            data = np.copy(demos[i])
            offset = - np.min(data, axis=0) / (np.max(data, axis=0) - np.min(data, axis=0)) - 0.5
            # data = data * scale + offset
            data = data * scale - np.mean(data * scale, axis=0)
            data = data - np.max(data, 0)
            data_mirror = np.flip(data.copy() * np.array([-1, 1]), 0)
            data = np.flip(data.copy(), 0)
            data = np.hstack((data, np.ones((data.shape[0], 1))))
            data = data / np.linalg.norm(data, axis=1)[:, None]
            data_mirror = np.hstack((data_mirror, np.ones((data.shape[0], 1))))
            data_mirror = data_mirror / np.linalg.norm(data_mirror, axis=1)[:, None]

            demos_on_sphere += [data]
            demos_on_sphere += [data_mirror]

        demos_on_sphere = np.array(demos_on_sphere)[:, ::subsampling, :]
        stacked_demos_on_sphere = np.vstack(demos_on_sphere)

        self.demos = torch.Tensor(demos_on_sphere.copy())
        self.data = torch.Tensor(stacked_demos_on_sphere)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DatasetRefCond(Dataset):
    def __init__(self, dataset, n_pred=1, n_ref=1, n_cond=1, w_cond=None):
        if n_cond > 1:
            raise NotImplementedError

        self.demos = dataset.demos
        self.dim = dataset.dim
        nb_demos = self.demos.shape[0]
        nb_samples = self.demos.shape[1]

        if n_pred > 1:
            manifolds = [(dataset.manifold, self.dim) for n in range(n_pred)]
            self.manifold = ProductManifoldTrajectories(*manifolds)
            # self.manifold = ProductManifold(*manifolds)
        else:
            self.manifold = dataset.manifold

        data = []
        for n in range(nb_demos):
            for i in range(n_ref + n_cond, nb_samples - n_pred):
                predictions = self.demos[n, i:i + n_pred, :].reshape(-1)
                reference = self.demos[n, i-n_ref:i, :].reshape(-1)
                if n_cond == 0:
                    data_i = torch.hstack((predictions, reference))[None]
                    data += [data_i]
                else:
                    if w_cond > 0:
                        start_w_cond = np.maximum(0, i - n_ref - w_cond)
                    else:
                        start_w_cond = 0
                    conditioning = torch.hstack((self.demos[n, start_w_cond:i - n_ref, :],
                                                 i - torch.range(start_w_cond, i - n_ref - 1)[:, None]))
                    n_cond_samples = i - n_ref - start_w_cond
                    data_i = torch.hstack((predictions.repeat([n_cond_samples, 1]), reference.repeat([n_cond_samples, 1]), conditioning))
                    # if noise:
                    #     conditionings += sigma + torch.randn_like(conditionings)

                    if w_cond > 0:
                        n_reps = int(w_cond / n_cond_samples)
                    else:
                        n_reps = int(nb_samples / (i - 1))

                    data += [data_i.repeat([n_reps, 1])]

        self.data = torch.cat(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DatasetVisionRefCond(Dataset):
    def __init__(self, dataset, image_demos, n_pred=1, n_ref=1, n_cond=1, w_cond=None):
        if n_cond > 1:
            raise NotImplementedError

        self.demos = dataset.demos
        self.dim = dataset.dim
        nb_demos = self.demos.shape[0]
        nb_samples = self.demos.shape[1]

        # We keep only one channel as we have greyscale images
        self.image_demos = image_demos[:, :, 0, :, :]

        if n_pred > 1:
            manifolds = [(dataset.manifold, self.dim) for n in range(n_pred)]
            self.manifold = ProductManifoldTrajectories(*manifolds)
        else:
            self.manifold = dataset.manifold

        data = []
        for n in range(nb_demos):
            for i in range(n_ref + n_cond, nb_samples - n_pred):
                predictions = self.demos[n, i:i + n_pred, :].reshape(-1)
                reference = self.image_demos[n, i-n_ref:i, :].reshape(-1)
                if n_cond == 0:
                    data_i = torch.hstack((predictions, reference))[None]
                    data += [data_i]
                else:
                    if w_cond > 0:
                        start_w_cond = np.maximum(0, i - n_ref - w_cond)
                    else:
                        start_w_cond = 0
                    n_cond_samples = i - n_ref - start_w_cond
                    conditioning_image = self.image_demos[n, start_w_cond:i - n_ref].reshape(n_cond_samples, -1)
                    conditioning_scalar = i - torch.range(start_w_cond, i - n_ref - 1)[:, None]
                    conditioning = torch.hstack((conditioning_image, conditioning_scalar))
                    data_i = torch.hstack((predictions.repeat([n_cond_samples, 1]), reference.repeat([n_cond_samples, 1]), conditioning))
                    # if noise:
                    #     conditionings += sigma + torch.randn_like(conditionings)
                    if w_cond > 0:
                        n_reps = int(w_cond / n_cond_samples)
                    else:
                        n_reps = int(nb_samples / (i - 1))

                    data += [data_i.repeat([n_reps, 1])]

        self.data = torch.cat(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class EuclideanPushtLowDimDatasetWrapper(PushTLowdimDataset):
    def __init__(self, zarr_path, horizon=1, pad_before=0, pad_after=0, obs_key='keypoint', state_key='state',
                 action_key='action', seed=42, val_ratio=0.0, max_train_episodes=None):
        super().__init__(zarr_path, horizon, pad_before, pad_after, obs_key, state_key, action_key, seed, val_ratio,
                         max_train_episodes)
        self.manifold = Euclidean()
        self.dim = 2


class EuclideanPushTStateDatasetDatasetWrapper(PushTStateDataset):
    def __init__(self, dataset_path, pred_horizon, obs_horizon, action_horizon):
        super().__init__(dataset_path, pred_horizon, obs_horizon, action_horizon)
        self.dim = 2
        manifolds = [(Euclidean(), self.dim) for n in range(pred_horizon)]
        self.manifold = ProductManifoldTrajectories(*manifolds)


class EuclideanVisionPushTStateDatasetDatasetWrapper(PushTImageDataset):
    def __init__(self, dataset_path, obs_horizon, pred_horizon=16):
        super().__init__(dataset_path, horizon=pred_horizon)
        self.dim = 2
        self.obs_horizon = obs_horizon
        manifolds = [(Euclidean(), self.dim) for n in range(pred_horizon)]
        self.manifold = ProductManifoldTrajectories(*manifolds)
        stats = dict()
        normalized_train_data = dict()
        for key, data in self.replay_buffer.items():
            if key == 'img':
                normalized_train_data[key] = data
            else:
                stats[key] = get_data_stats(data)
                normalized_train_data[key] = normalize_data(data, stats[key])
        # self.normalized_train_data = normalized_train_data
        self.stats = stats
        self.sampler.replay_buffer = normalized_train_data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        torch_data['obs']['agent_pos'] = torch_data['obs']['agent_pos'][:self.obs_horizon]
        torch_data['obs']['image'] = torch_data['obs']['image'][:self.obs_horizon]
        return torch_data


class EuclideanVisionPushTStateDatasetDatasetRefCondWrapper(PushTImageDataset):
    def __init__(self, dataset_path, obs_horizon, pred_horizon=16, condition_band=16):
        super().__init__(dataset_path, horizon=pred_horizon + condition_band)
        self.dim = 2
        self.obs_horizon = obs_horizon
        manifolds = [(Euclidean(), self.dim) for n in range(pred_horizon)]
        self.manifold = ProductManifoldTrajectories(*manifolds)
        stats = dict()
        normalized_train_data = dict()
        for key, data in self.replay_buffer.items():
            if key == 'img':
                normalized_train_data[key] = data
            else:
                stats[key] = get_data_stats(data)
                normalized_train_data[key] = normalize_data(data, stats[key])
        # self.normalized_train_data = normalized_train_data
        self.stats = stats
        self.sampler.replay_buffer = normalized_train_data
        self.condition_band = condition_band

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        torch_data['action'] = torch_data['action'][self.condition_band:]
        cond_index = torch.randint(self.condition_band, (1,))
        cond_img = torch_data['obs']['image'][cond_index]
        ref_img = torch_data['obs']['image'][self.condition_band:self.condition_band + self.obs_horizon]
        torch_data['obs']['image'] = torch.cat([cond_img, ref_img])

        ref_agent_pos = torch_data['obs']['agent_pos'][cond_index]
        cond_agent_pos = torch_data['obs']['agent_pos'][self.condition_band:self.condition_band + self.obs_horizon]
        torch_data['obs']['agent_pos'] = torch.cat([cond_agent_pos, ref_agent_pos])
        return torch_data


class SphereVisionPushTStateDatasetDatasetWrapper(PushTImageDataset):
    def __init__(self, dataset_path, obs_horizon, pred_horizon=16):
        super().__init__(dataset_path, horizon=pred_horizon)
        self.dim = 3
        self.obs_horizon = obs_horizon
        manifolds = [(Sphere(), self.dim) for n in range(pred_horizon)]
        self.manifold = ProductManifoldTrajectories(*manifolds)
        stats = dict()
        # load sphere dataset
        sp_img = np.load(dataset_path + '/../sphere_dataset/sphere_img.npy')
        sp_agent_pos = np.load(dataset_path + '/../sphere_dataset/sphere_agent_pos.npy')
        sp_action = np.load(dataset_path + '/../sphere_dataset/sphere_action.npy')
        normalized_sp_data = dict()
        normalized_sp_data['img'] = sp_img
        normalized_sp_data['action'] = sp_action
        normalized_sp_data['state'] = sp_agent_pos

        self.stats = stats
        self.sampler.replay_buffer = normalized_sp_data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        torch_data['obs']['agent_pos'] = torch_data['obs']['agent_pos'][:self.obs_horizon]
        torch_data['obs']['image'] = torch_data['obs']['image'][:self.obs_horizon]
        return torch_data


# the difference of this class to the above one is that the previous one, observation
# are continuous; but in this one, the observation can be uncontinuous, that is with one reference
# one step before the current step, and one conditional point in certain range before the current step
class SphereVisionPushTStateDatasetDatasetRefCondWrapper(PushTImageDataset):
    def __init__(self, dataset_path, obs_horizon, pred_horizon=16, condition_band=16):
        super().__init__(dataset_path, horizon=pred_horizon + condition_band)
        self.dim = 3
        self.obs_horizon = obs_horizon
        manifolds = [(Sphere(), self.dim) for n in range(pred_horizon)]
        self.manifold = ProductManifoldTrajectories(*manifolds)
        stats = dict()
        # load sphere dataset
        sp_img = np.load(dataset_path + '/../sphere_dataset/sphere_img.npy')
        sp_agent_pos = np.load(dataset_path + '/../sphere_dataset/sphere_agent_pos.npy')
        sp_action = np.load(dataset_path + '/../sphere_dataset/sphere_action.npy')
        normalized_sp_data = dict()
        normalized_sp_data['img'] = sp_img
        normalized_sp_data['action'] = sp_action
        normalized_sp_data['state'] = sp_agent_pos

        self.stats = stats
        self.sampler.replay_buffer = normalized_sp_data
        self.condition_band = condition_band

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)

        torch_data['action'] = torch_data['action'][self.condition_band:]
        cond_index = torch.randint(self.condition_band, (1,))
        cond_img = torch_data['obs']['image'][cond_index]
        ref_img = torch_data['obs']['image'][self.condition_band:self.condition_band + self.obs_horizon]
        torch_data['obs']['image'] = torch.cat([cond_img, ref_img])

        ref_agent_pos = torch_data['obs']['agent_pos'][cond_index]
        cond_agent_pos = torch_data['obs']['agent_pos'][self.condition_band:self.condition_band + self.obs_horizon]
        torch_data['obs']['agent_pos'] = torch.cat([cond_agent_pos, ref_agent_pos])
        return torch_data


# todo generate a mixed manifold
# dont forget to do the normalization
class RealRobotVisionDishGraspStateDatasetRefCond(torch.utils.data.Dataset):
    def __init__(self, dataset_path, obs_horizon, pred_horizon=16):
        super.__init__()
        self.dim = 7  # 3 for position, 4 for quaternion
        self.pos_dim = 3
        self.quaternion_dim = 4
        manifolds = []
        for _ in range(pred_horizon):
            manifolds.append((Euclidean(), self.pos_dim))
            manifolds.append((Sphere(), self.quaternion_dim))
        self.manifold = ProductManifoldTrajectories(*manifolds)
        # todo read the iamge, the action & position seriers


class RealRobotVisionDishGraspEuclideanStateDatasetRefCond(torch.utils.data.Dataset):
    def __init__(self, dataset_path, obs_horizon, pred_horizon=16):
        super.__init__()
        self.dim = 7  # 3 for position, 4 for quaternion
        self.pos_dim = 3
        self.quaternion_dim = 4
        manifolds = []
        manifolds = [(Euclidean(), self.dim) for n in range(pred_horizon)]
        self.manifold = ProductManifoldTrajectories(*manifolds)
        # todo read the iamge, the action & position seriers


def _get_dataset(cfg):
    expand_factor = 1
    if cfg.data == "volcano":
        dataset = Volcano(cfg.get("earth_datadir", cfg.get("datadir", None)))
        expand_factor = 1550
    elif cfg.data == "earthquake":
        dataset = Earthquake(cfg.get("earth_datadir", cfg.get("datadir", None)))
        expand_factor = 210
    elif cfg.data == "fire":
        dataset = Fire(cfg.get("earth_datadir", cfg.get("datadir", None)))
        expand_factor = 100
    elif cfg.data == "flood":
        dataset = Flood(cfg.get("earth_datadir", cfg.get("datadir", None)))
        expand_factor = 260
    elif cfg.data == "general":
        dataset = Top500(cfg.top500_datadir, amino="General")
        expand_factor = 1
    elif cfg.data == "glycine":
        dataset = Top500(cfg.top500_datadir, amino="Glycine")
        expand_factor = 10
    elif cfg.data == "proline":
        dataset = Top500(cfg.top500_datadir, amino="Proline")
        expand_factor = 18
    elif cfg.data == "prepro":
        dataset = Top500(cfg.top500_datadir, amino="Pre-Pro")
        expand_factor = 20
    elif cfg.data == "rna":
        dataset = RNA(cfg.rna_datadir)
        expand_factor = 14
    elif cfg.data == "simple_bunny":
        dataset = SimpleBunny(cfg.mesh_datadir)
    elif cfg.data == "bunny10":
        dataset = Bunny10(cfg.mesh_datadir)
    elif cfg.data == "bunny50":
        dataset = Bunny50(cfg.mesh_datadir)
    elif cfg.data == "bunny100":
        dataset = Bunny100(cfg.mesh_datadir)
    elif cfg.data == "spot10":
        dataset = Spot10(cfg.mesh_datadir)
    elif cfg.data == "spot50":
        dataset = Spot50(cfg.mesh_datadir)
    elif cfg.data == "spot100":
        dataset = Spot100(cfg.mesh_datadir)
    elif cfg.data == "maze3v2":
        dataset = Maze3v2(cfg.mesh_datadir)
    elif cfg.data == "maze4v2":
        dataset = Maze4v2(cfg.mesh_datadir)
    elif cfg.data == "wrapped_torus":
        manifold = FlatTorus()
        dataset = Wrapped(
            manifold,
            cfg.wrapped.dim,
            cfg.wrapped.n_mixtures,
            cfg.wrapped.scale,
            dataset_size=200000,
        )
    elif cfg.data == "wrapped_spd":
        manifold = SPD()
        d = cfg.wrapped.dim
        n = manifold.matdim(d)
        manifold = SPD(scale_std=0.5, scale_Id=3.0, base_expmap=False)
        centers = manifold.vectorize(torch.eye(n) * 2.0).reshape(1, -1)

        dataset = Wrapped(
            manifold,
            cfg.wrapped.dim,
            cfg.wrapped.n_mixtures,
            cfg.wrapped.scale,
            centers=centers,
            dataset_size=10000,
        )
    # elif cfg.data == "eeg_1":
    #     dataset = EEG(cfg.eeg_datadir, set="1", Riem_geodesic=cfg.eeg.Riem_geodesic, Riem_norm=cfg.eeg.Riem_norm)
    # elif cfg.data == "eeg_2a":
    #     dataset = EEG(cfg.eeg_datadir, set="2a", Riem_geodesic=cfg.eeg.Riem_geodesic, Riem_norm=cfg.eeg.Riem_norm)
    # elif cfg.data == "eeg_2b":
    #     dataset = EEG(cfg.eeg_datadir, set="2b", Riem_geodesic=cfg.eeg.Riem_geodesic, Riem_norm=cfg.eeg.Riem_norm)
    elif cfg.data == "hyperbolic":
        dataset = HyperbolicDatasetPair()
    elif cfg.data == "lasa_euclidean":
        dataset = EuclideanLasaDataset(cfg.get("datadir"), cfg.get("letter"))
        expand_factor = 100
    elif cfg.data == "lasa_euclidean_multimodal":
        dataset = EuclideanMultimodalDataset(cfg.get("datadir"))
        expand_factor = 100
    elif cfg.data == "lasa_sphere":
        dataset = SphereLasaDataset(cfg.get("datadir"), cfg.get("letter"))
        expand_factor = 100
    elif cfg.data == "lasa_sphere_multimodal":
        dataset = SphereMultimodalDataset(cfg.get("datadir"))
        expand_factor = 100
    elif cfg.data == "lasa_euclidean_ref_cond":
        dataset = DatasetRefCond(EuclideanLasaDataset(cfg.get("datadir"), cfg.get("letter")), n_pred=cfg.get("n_pred"),
                                 n_ref=cfg.get("n_ref"), n_cond=cfg.get("n_cond"), w_cond=cfg.get("w_cond"))
        if cfg.get("n_cond") == 0:
            expand_factor = 100
        elif cfg.get("w_cond") > 0:
            expand_factor = int(200 / cfg.get("w_cond"))
    elif cfg.data == "lasa_euclidean_multimodal_ref_cond":
        dataset = DatasetRefCond(EuclideanMultimodalDataset(cfg.get("datadir")), n_pred=cfg.get("n_pred"),
                                 n_ref=cfg.get("n_ref"), n_cond=cfg.get("n_cond"), w_cond=cfg.get("w_cond"))
        if cfg.get("n_cond") == 0:
            expand_factor = 100
        elif cfg.get("w_cond") > 0:
            expand_factor = int(200 / cfg.get("w_cond"))
    elif cfg.data == "lasa_sphere_ref_cond":
        dataset = DatasetRefCond(SphereLasaDataset(cfg.get("datadir"), cfg.get("letter")), n_pred=cfg.get("n_pred"),
                                 n_ref=cfg.get("n_ref"), n_cond=cfg.get("n_cond"), w_cond=cfg.get("w_cond"))
        if cfg.get("n_cond") == 0:
            expand_factor = 100
        elif cfg.get("w_cond") > 0:
            expand_factor = int(200 / cfg.get("w_cond"))
    elif cfg.data == "lasa_sphere_multimodal_ref_cond":
        dataset = DatasetRefCond(SphereMultimodalDataset(cfg.get("datadir")), n_pred=cfg.get("n_pred"),
                                 n_ref=cfg.get("n_ref"), n_cond=cfg.get("n_cond"), w_cond=cfg.get("w_cond"))
        if cfg.get("n_cond") == 0:
            expand_factor = 100
        elif cfg.get("w_cond") > 0:
            expand_factor = int(200 / cfg.get("w_cond"))
    elif cfg.data == "lasa_euclidean_vision_ref_cond":
        image_demos = np.load(os.path.join(cfg.get("image_datadir"), cfg.get("letter") + '.npy')).squeeze()
        if image_demos.shape[-1] == 3:
            image_demos = np.swapaxes(np.swapaxes(image_demos, -2, -1), -3, -2)
        image_demos = torch.Tensor(image_demos)
        dataset = DatasetVisionRefCond(EuclideanLasaDataset(cfg.get("datadir"), cfg.get("letter")),
                                       image_demos=image_demos, n_pred=cfg.get("n_pred"), n_ref=cfg.get("n_ref"),
                                       n_cond=cfg.get("n_cond"), w_cond=cfg.get("w_cond"))
        if cfg.get("n_cond") == 0:
            expand_factor = 100
        elif cfg.get("w_cond") > 0:
            expand_factor = int(200 / cfg.get("w_cond"))
    elif cfg.data == "lasa_sphere_vision_ref_cond":
        image_demos = np.load(os.path.join(cfg.get("image_datadir"), cfg.get("letter") + '.npy')).squeeze()
        if image_demos.shape[-1] == 3:
            image_demos = np.swapaxes(np.swapaxes(image_demos, -2, -1), -3, -2)
        image_demos = torch.Tensor(image_demos)
        dataset = DatasetVisionRefCond(SphereLasaDataset(cfg.get("datadir"), cfg.get("letter")),
                                       image_demos=image_demos, n_pred=cfg.get("n_pred"), n_ref=cfg.get("n_ref"),
                                       n_cond=cfg.get("n_cond"), w_cond=cfg.get("w_cond"))
        if cfg.get("n_cond") == 0:
            expand_factor = 100
        elif cfg.get("w_cond") > 0:
            expand_factor = int(200 / cfg.get("w_cond"))

    elif cfg.data == "pusht_euclidean_ref_cond":
        dataset = EuclideanPushTStateDatasetDatasetWrapper(cfg.get("datadir") + "/pusht_cchi_v7_replay.zarr",
                                                           pred_horizon=16, obs_horizon=2, action_horizon=8)

        # dataset = EuclideanPushtLowDimDatasetWrapper(horizon=16, max_train_episodes=90, pad_after=7, pad_before=1,
        #                                              seed=42, val_ratio=0.02,
        #                                              zarr_path=cfg.get("datadir") + "/pusht_cchi_v7_replay.zarr")
    elif cfg.data == 'pusht_vision_ref_cond':
        dataset = EuclideanVisionPushTStateDatasetDatasetWrapper(cfg.get("datadir") + "/pusht_cchi_v7_replay.zarr",
                                                           obs_horizon=2)
    elif cfg.data == 'pusht_vision_ref_cond_band':
        dataset = EuclideanVisionPushTStateDatasetDatasetRefCondWrapper(cfg.get("datadir") + "/pusht_cchi_v7_replay.zarr",
                                                           obs_horizon=cfg.n_ref)
    elif cfg.data == 'pusht_sphere_vision_ref_cond':
        dataset = SphereVisionPushTStateDatasetDatasetWrapper(cfg.get("datadir") + "/pusht_cchi_v7_replay.zarr",
                                                                 obs_horizon=2)
    elif cfg.data == 'pusht_sphere_vision_ref_cond_band':
        dataset = SphereVisionPushTStateDatasetDatasetRefCondWrapper(cfg.get("datadir") + "/pusht_cchi_v7_replay.zarr",
                                                                 obs_horizon=cfg.n_ref)
    else:
        raise ValueError("Unknown dataset option '{name}'")
    return dataset, expand_factor


def get_loaders(cfg):
    dataset, expand_factor = _get_dataset(cfg)

    N = len(dataset)
    N_val = N_test = N // 10
    N_train = N - N_val - N_test

    data_seed = cfg.seed if cfg.data_seed is None else cfg.data_seed
    if data_seed is None:
        raise ValueError("seed for data generation must be provided")
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset,
        [N_train, N_val, N_test],
        generator=torch.Generator().manual_seed(data_seed),
    )

    # Expand the training set (we optimize based on number of iterations anyway).
    train_set = ExpandDataset(train_set, expand_factor=expand_factor)

    train_loader = DataLoader(
        train_set, cfg.optim.batch_size, shuffle=True, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_set, cfg.optim.val_batch_size, shuffle=False, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, cfg.optim.val_batch_size, shuffle=False, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def get_manifold(cfg):
    dataset, _ = _get_dataset(cfg)

    if isinstance(dataset, MeshDataset) or isinstance(dataset, MeshDatasetPair):
        manifold = dataset.manifold(
            numeigs=cfg.mesh.numeigs, metric=Metric(cfg.mesh.metric), temp=cfg.mesh.temp
        )
        return manifold, dataset.dim
    else:
        return dataset.manifold, dataset.dim

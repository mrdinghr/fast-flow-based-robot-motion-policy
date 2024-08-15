"""Copyright (c) Meta Platforms, Inc. and affiliates."""
import copy

import torch
import torch.nn.functional as F
from torch.func import vjp, jvp, vmap, jacrev
from torchdiffeq import odeint

from manifm.ema import EMA
from manifm.model.arch import tMLP, ProjectToTangent, Unbatch
from manifm.manifolds import (
    Sphere,
    FlatTorus,
    Euclidean,
    ProductManifold,
    Mesh,
    SPD,
    PoincareBall,
)
from manifm.manifolds import geodesic
from manifm.solvers import projx_integrator_return_last, projx_integrator, projx_integrator_guided
from manifm.model_pl import div_fn, output_and_div, ManifoldFMLitModule
from manifm.vision.resnet_models import get_resnet, replace_bn_with_gn
from manifm.model.uNet import Unet


class ManifoldVisionTrajectoriesMugRotateFMLitModule(ManifoldFMLitModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.n_pred = cfg.n_pred
        self.n_ref = cfg.n_ref
        self.n_cond = cfg.n_cond
        self.w_cond = cfg.w_cond
        if self.n_cond > 0:
            add_dim = 1
        else:
            add_dim = 0

        # Vision model
        self.vision_encoder = get_resnet('resnet18')
        self.vision_encoder = replace_bn_with_gn(self.vision_encoder)
        # Dimensions
        self.image_dim = cfg.image_dim
        # self.dim 8: 3 + 4 + 1  pos  quaternion  grip
        if cfg.use_gripper_img:
            self.vision_feature_dim = 512 * 2 + self.dim
        else:
            self.vision_feature_dim = 512 + self.dim

        # ResNet18 has output dim of 512
        self.n_pred = cfg.n_pred
        self.output_dim = self.dim * self.n_pred
        self.input_dim = self.output_dim + \
                         self.n_ref * self.vision_feature_dim + \
                         self.n_cond * self.vision_feature_dim + add_dim  # n x x_t (n x 3) + 1x ref (3) + 1x cond (3+1)

        self.model_type = cfg.model_type
        # Redefined the model of the vector field.
        if cfg.model_type == 'Unet':
            self.model = Unbatch(  # Ensures vmap works.
                ProjectToTangent(  # Ensures we can just use Euclidean divergence.
                    Unet(input_dim=self.dim,
                         global_cond_dim=self.n_ref * self.vision_feature_dim + \
                                         self.n_cond * self.vision_feature_dim + add_dim,
                         # down_dims=[512, 1024, 2048]),
                         # down_dims=[256, 512, 1024]),
                         down_dims=[128, 256, 512]),
                    manifold=self.manifold,
                    metric_normalize=self.cfg.model.get("metric_normalize", False),
                    dim=self.output_dim  # As the last dims of x are the conditioning variable
                )
            )
        elif cfg.model_type == 'tMLP':
            self.model = EMA(
                Unbatch(  # Ensures vmap works.
                    ProjectToTangent(  # Ensures we can just use Euclidean divergence.
                        tMLP(  # Vector field in the ambient space.
                            self.input_dim,
                            d_out=self.output_dim,
                            d_model=cfg.model.d_model,
                            num_layers=cfg.model.num_layers,
                            actfn=cfg.model.actfn,
                            fourier=cfg.model.get("fourier", None),
                        ),
                        manifold=self.manifold,
                        metric_normalize=self.cfg.model.get("metric_normalize", False),
                        dim=self.output_dim  # As the last dims of x are the conditioning variable
                    )
                ),
                cfg.optim.ema_decay,
            )
        else:
            assert False, 'model type wrong'

        self.cfg = cfg
        self.small_var = False

        # used for normalization of pos & quaternion
        # self.pos_quat_max = torch.tensor([0.7, 0.3, 0.4, 1., 1., 1., 1.]).to(self.device)
        # self.pos_quat_min = torch.tensor([0.2, -0.1, 0., -1., -1., -1., -1.]).to(self.device)

        # new simple dataset best for 25 demos
        self.pos_quat_max = torch.tensor([0.6, 0.25, 0.4, 1., 1., 1., 1.]).to(self.device)
        self.pos_quat_min = torch.tensor([0.2, 0., 0.2, -1., -1., -1., -1.]).to(self.device)

        self.normalize = cfg.normalize_pos_quat
        self.nets = EMA(torch.nn.ModuleDict({'vision_encoder': self.vision_encoder, 'vf_net': self.model}))

    @property
    def vecfield(self):
        return self.model

    @torch.no_grad()
    def sample_all(self, n_samples, device, xref, xcond=None, x0=None, different_sample=False, ode_steps=11,
                   guided_flow=False):
        if self.small_var and x0 is not None:
            x0 = x0.reshape((n_samples, self.cfg.n_pred, self.dim))
            x0[:, :, 0:3] *= 0.05
            x0[:, :, -1] *= 0.05
            x0 = x0.reshape((n_samples, self.cfg.n_pred * self.dim))
        if x0 is None:
            # Sample from base distribution.
            sample = self.manifold.random_base(n_samples, self.output_dim, different_sample=different_sample)
            sample = sample.reshape((n_samples, self.cfg.n_pred, self.dim))
            if self.small_var:
                # sample[:, :, 0:3] *= 0.05
                # sample[:, :, -1] *= 0.05
                sample[:, :, 0:3] *= 0.00
                sample[:, :, -1] *= 0.00
                sample = sample.reshape((n_samples, self.cfg.n_pred * self.dim))
            # x0 = (
            #     self.manifold.random_base(n_samples, self.output_dim, different_sample=different_sample)
            #     .reshape(n_samples, self.output_dim)
            #     .to(device)
            # )
            x0 = (sample.reshape((n_samples, self.output_dim)).to(device))

        if self.model_type == 'Unet':
            if xcond is not None:
                if self.cfg.model_type == 'Unet':
                    self.model.vecfield.vecfield.unet.global_cond = torch.hstack((xref, xcond))
                else:
                    print('wrong cfg model_type setting')
            else:
                if self.cfg.model_type == 'Unet':
                    self.model.vecfield.vecfield.unet.global_cond = xref
                else:
                    assert False, 'MODEL TYPE WRONG SETTING'
            if not guided_flow:
                xs, _ = projx_integrator(
                    self.manifold,
                    self.vecfield,
                    x0,
                    t=torch.linspace(0, 1, ode_steps).to(device),
                    method="euler",
                    projx=False,
                    pbar=True,
                    local_coords=True,
                )
            else:
                xs, _ = projx_integrator_guided(
                    self.manifold,
                    self.vecfield,
                    x0,
                    t=torch.linspace(0, 1, ode_steps).to(device),
                    method="euler",
                    projx=True,
                    pbar=True,
                    xref=xref
                )
        else:
            # Wrapper for conditioning
            wrapper_vecfield = self.wrapper_red_cond(self.vecfield, xref, xcond)

            # Solve ODE.
            xs, _ = projx_integrator(
                self.manifold,
                wrapper_vecfield,
                x0,
                t=torch.linspace(0, 1, 1001).to(device),
                method="euler",
                projx=True,
                pbar=True,
            )
        return xs

    def normalize_pos_quat(self, pos_quat):
        return 2 * (pos_quat - self.pos_quat_min.to(self.device)) / (
                self.pos_quat_max.to(self.device) - self.pos_quat_min.to(self.device)) - 1

    def denormalize_pos_quat(self, norm_pos_quat):
        return (norm_pos_quat + 1) / 2 * (
                self.pos_quat_max.to(self.device) - self.pos_quat_min.to(self.device)) + self.pos_quat_min.to(
            self.device)

    def normalize_grip(self, grip_state):
        return grip_state / 0.03

    def denormalize_grip(self, norm_grip_state):
        return norm_grip_state * 0.03

    def image_to_features(self, x):
        image_feaeture = self.vision_encoder(x.flatten(end_dim=1))
        image_feaeture = image_feaeture.reshape(*x.shape[:2], -1)
        return image_feaeture.flatten(start_dim=1)

    # todo this function is used for removing batch that robto doesnt move
    def check_batch_no_move(self, batch):
        pos_quat_actions = batch['traj']['target_pos_quat']['action'].to(self.device)
        move_distance = pos_quat_actions - pos_quat_actions[:, 0:1, :]
        move_distance = move_distance.abs().sum(dim=(1, 2))
        move_batch_index = torch.where(move_distance > 5e-2)[0].tolist()
        new_batch = copy.deepcopy(batch)
        new_batch['observation']['v_fix'] = batch['observation']['v_fix'][move_batch_index]
        new_batch['traj']['target_pos_quat']['action'] = batch['traj']['target_pos_quat']['action'][move_batch_index]
        new_batch['traj']['target_pos_quat']['obs'] = batch['traj']['target_pos_quat']['obs'][move_batch_index]
        new_batch['traj']['gripper']['obs'] = batch['traj']['gripper']['obs'][move_batch_index]
        new_batch['traj']['gripper']['action'] = batch['traj']['gripper']['action'][move_batch_index]
        return new_batch

    def unpack_predictions_reference_conditioning_samples(self, batch: torch.Tensor):
        # Unpack batch vector into different components
        # Assumes greyscale images
        # if self.del_unmove_batch:
        # batch = self.check_batch_no_move(batch)
        B = batch['observation']['v_fix'].shape[0]

        x1_pos_quat = batch['traj']['target_pos_quat']['action'].to(self.device)
        xref_pos_quat = batch['traj']['target_pos_quat']['obs'].to(self.device).float()
        xref_gripper = batch['traj']['gripper']['obs'].to(self.device)
        # make sure x1[0] == xref[-1]
        x1_pos_quat = torch.cat([xref_pos_quat[:, -1:, :], x1_pos_quat], dim=1)
        # x1_pos_quat

        # regulate the quaternion, because like  0 1 0 0 and 0 -1 0 0 are same rotation. this line is to make sure all rotation in same way
        x1_pos_quat[..., 3:] *= torch.sign(x1_pos_quat[..., 4]).unsqueeze(-1)
        x1_gripper = batch['traj']['gripper']['action'].unsqueeze(-1).to(self.device)
        x1_gripper = torch.cat([xref_gripper[..., -1].unsqueeze(-1).unsqueeze(-1), x1_gripper], dim=1)
        if self.normalize:
            x1_pos_quat = self.normalize_pos_quat(x1_pos_quat)
        x1 = torch.cat((x1_pos_quat, x1_gripper), dim=2).reshape((B, self.n_pred * self.dim)).float()
        x0 = self.manifold.random_base(x1.shape[0], self.output_dim).to(x1)

        # image reference scalar: fix_img
        fix_img = batch['observation']['v_fix'].to(self.device)  # shape B * n_ref * 3 * H * W
        fix_img_feature = self.image_to_features(fix_img)
        img_scalar = fix_img_feature.reshape((B, self.n_ref * 512))

        xref_pos_quat[..., 3:] *= torch.sign(xref_pos_quat[..., 4]).unsqueeze(-1)
        if self.normalize:
            xref_pos_quat = self.normalize_pos_quat(xref_pos_quat)
        xref_pos_quat = xref_pos_quat.reshape((B, self.n_ref * (self.dim - 1)))
        xref = torch.cat((img_scalar, xref_pos_quat, xref_gripper), dim=1).float()
        return x1, xref, None, x0

    class wrapper_red_cond(torch.nn.Module):
        """Wraps model to torchdyn compatible format."""

        def __init__(self, vecfield, ref, cond):
            super().__init__()
            self.vecfield = vecfield
            self.ref = ref
            self.cond = cond

        def forward(self, t, x):
            return self.vecfield(t, torch.hstack((x, self.ref, self.cond)))

    def rfm_loss_fn(self, batch: torch.Tensor):
        x1, xref, xcond, x0 = self.unpack_predictions_reference_conditioning_samples(batch)

        N = x1.shape[0]
        t = torch.rand(N).reshape(-1, 1).to(x1)

        def cond_u(x0, x1, t):
            path = geodesic(self.manifold, x0, x1)
            x_t, u_t = jvp(path, (t,), (torch.ones_like(t).to(t),))
            return x_t, u_t

        x_t, u_t = vmap(cond_u)(x0, x1, t)
        x_t = x_t.reshape(N, self.output_dim)
        u_t = u_t.reshape(N, self.output_dim)

        if self.model_type == 'Unet':
            if xcond is not None:
                self.model.vecfield.vecfield.unet.global_cond = torch.hstack((xref, xcond))
            else:
                self.model.vecfield.vecfield.unet.global_cond = xref

            diff = self.vecfield(t, x_t) - u_t
        else:
            if xcond is not None:
                x_t_ref_cond = torch.hstack((x_t, xref, xcond))
            else:
                x_t_ref_cond = xref
            diff = self.vecfield(t, x_t_ref_cond) - u_t
        return self.manifold.inner(x_t, diff, diff).mean() / self.output_dim

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.nets.parameters(),
            lr=self.cfg.optim.lr,
            weight_decay=self.cfg.optim.wd,
            eps=self.cfg.optim.eps,
        )

        if self.cfg.optim.get("scheduler", "cosine") == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.cfg.optim.num_iterations,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        else:
            return {
                "optimizer": optimizer,
            }

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if isinstance(self.nets, EMA):
            self.nets.update_ema()

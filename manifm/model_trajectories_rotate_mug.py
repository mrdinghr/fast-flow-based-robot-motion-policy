"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import torch
import torch.nn.functional as F
from torch.func import vjp, jvp, vmap, jacrev
from torchdiffeq import odeint

from manifm.ema import EMA
from manifm.model.arch import tMLP, ProjectToTangent, Unbatch

from manifm.manifolds import geodesic
from manifm.solvers import projx_integrator_return_last, projx_integrator
from manifm.model_pl import div_fn, output_and_div, ManifoldFMLitModule
from manifm.model.uNet import Unet


'''
class for RFMP roate mug with state based obsrevation

Note: not used any more
'''


class ManifoldStateTrajectoriesMugRotateFMLitModule(ManifoldFMLitModule):
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

        # ResNet18 has output dim of 512
        self.n_pred = cfg.n_pred
        self.output_dim = self.dim * self.n_pred
        self.input_dim = self.output_dim + \
                         self.n_ref * self.dim + \
                         self.n_cond * self.dim + add_dim  # n x x_t (n x 3) + 1x ref (3) + 1x cond (3+1)

        self.model_type = cfg.model_type
        # Redefined the model of the vector field.
        if cfg.model_type == 'Unet':
            self.model = EMA(
                Unbatch(  # Ensures vmap works.
                    ProjectToTangent(  # Ensures we can just use Euclidean divergence.
                        Unet(input_dim=self.dim,
                             global_cond_dim=self.n_ref * self.dim + \
                                             self.n_cond * self.dim + add_dim,
                             # down_dims=[512, 1024, 2048]),
                             # down_dims=[256, 512, 1024]),
                             down_dims=[128, 256, 512]),
                        manifold=self.manifold,
                        metric_normalize=self.cfg.model.get("metric_normalize", False),
                        dim=self.output_dim  # As the last dims of x are the conditioning variable
                    )
                ),
                cfg.optim.ema_decay,
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
        self.pos_quat_max = torch.tensor([0.7, 0.3, 0.4, 1., 1., 1., 1.]).to(self.device)
        self.pos_quat_min = torch.tensor([0.2, -0.1, 0., -1., -1., -1., -1.]).to(self.device)
        self.normalize = cfg.normalize_pos_quat


    @property
    def vecfield(self):
        return self.model

    @torch.no_grad()
    def sample_all(self, n_samples, device, xref, xcond=None, x0=None, different_sample=False, ode_steps=11):
        if self.small_var and x0 is not None:
            x0 = x0.reshape((n_samples, self.cfg.n_pred, self.dim))
            x0[:, :, 0:3] *= 0.05
            x0[:, :, -1] *= 0.05
            x0 = x0.reshape((n_samples, self.cfg.n_pred * self.dim))
        if x0 is None:
            # Sample from prior distribution.
            sample = self.manifold.random_base(n_samples, self.output_dim, different_sample=different_sample)
            sample = sample.reshape((n_samples, self.cfg.n_pred, self.dim))
            if self.small_var:
                sample[:, :, 0:3] *= 0.05
                sample[:, :, -1] *= 0.05
                sample = sample.reshape((n_samples, self.cfg.n_pred * self.dim))
            x0 = (sample.reshape((n_samples, self.output_dim)).to(device))

        if self.model_type == 'Unet' or self.model_type == 'Unet_noema':
            if xcond is not None:
                if self.cfg.model_type == 'Unet':
                    self.model.model.vecfield.vecfield.unet.global_cond = torch.hstack((xref, xcond))
                elif self.cfg.model_type == 'Unet_noema':
                    self.model.vecfield.vecfield.unet.global_cond = torch.hstack((xref, xcond))
                else:
                    print('wrong cfg model_type setting')
            else:
                if self.cfg.model_type == 'Unet':
                    self.model.model.vecfield.vecfield.unet.global_cond = xref
                elif self.cfg.model_type == 'Unet_noema':
                    self.model.vecfield.vecfield.unet.global_cond = xref
            xs, _ = projx_integrator(
                self.manifold,
                self.vecfield,
                x0,
                t=torch.linspace(0, 1, ode_steps).to(device),
                method="euler",
                projx=True,
                pbar=True,
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

    # during training, normalize data to [-1, 1]
    def normalize_pos_quat(self, pos_quat):
        return 2 * (pos_quat - self.pos_quat_min.to(self.device)) / (
                self.pos_quat_max.to(self.device) - self.pos_quat_min.to(self.device)) - 1

    # during testing, denormalize data to [-1, 1]
    def denormalize_pos_quat(self, norm_pos_quat):
        return (norm_pos_quat + 1) / 2 * (
                self.pos_quat_max.to(self.device) - self.pos_quat_min.to(self.device)) + self.pos_quat_min.to(
            self.device)

    # normalizing grip state to [0, 1] during training
    def normalize_grip(self, grip_state):
        return grip_state / 0.035

    # denormalizing grip state during testing
    def denormalize_grip(self, norm_grip_state):
        return norm_grip_state * 0.035

    # get xref from batch data
    def unpack_predictions_reference_conditioning_samples(self, batch: torch.Tensor):
        # Unpack batch vector into different components
        # Assumes greyscale images
        B = batch['observation']['v_fix'].shape[0]

        x1_pos_quat = batch['future_pos_quat'].to(self.device)
        x1_gripper = batch['future_gripper'].to(self.device)
        if self.normalize:
            x1_pos_quat = self.normalize_pos_quat(x1_pos_quat)
            x1_gripper = self.normalize_grip(x1_gripper)
        x1 = torch.cat((x1_pos_quat, x1_gripper), dim=2).reshape((B, self.n_pred * self.dim))
        x0 = self.manifold.random_base(x1.shape[0], self.output_dim).to(x1)

        # get xref for time step t - 1 and t
        xref_pos_quat_pre = batch['previous_pos_quat'].squeeze(dim=1).to(self.device)
        xref_pos_quat_cur = batch['future_pos_quat'][:, 0, :].to(self.device)
        if self.normalize:
            xref_pos_quat_pre = self.normalize_pos_quat(xref_pos_quat_pre)
            xref_pos_quat_cur = self.normalize_pos_quat(xref_pos_quat_cur)
        xref_pos_quat = torch.cat((xref_pos_quat_pre, xref_pos_quat_cur), dim=1)
        xref_gripper_pre = batch['previous_gripper'].squeeze(dim=1).to(self.device)
        xref_gripper_cur = batch['smooth_future_gripper'][:, 0, :].to(self.device)
        if self.normalize:
            xref_gripper_pre = self.normalize_grip(xref_gripper_pre)
            xref_gripper_cur = self.normalize_grip(xref_gripper_cur)
        xref_gripper = torch.cat((xref_gripper_pre, xref_gripper_cur), dim=1)  # shape B * 2
        xref = torch.cat((xref_pos_quat, xref_gripper), dim=1)
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

        if self.model_type == 'Unet' or self.model_type == 'Unet_noema':
            # set the observation and time step condition vector
            if xcond is not None:
                if self.cfg.model_type == 'Unet':
                    self.model.model.vecfield.vecfield.unet.global_cond = torch.hstack((xref, xcond))
                elif self.cfg.model_type == 'Unet_noema':
                    self.model.vecfield.vecfield.unet.global_cond = torch.hstack((xref, xcond))
            else:
                if self.cfg.model_type == 'Unet':
                    self.model.model.vecfield.vecfield.unet.global_cond =xref
                elif self.cfg.model_type == 'Unet_noema':
                    self.model.vecfield.vecfield.unet.global_cond = xref
            diff = self.vecfield(t, x_t) - u_t
        else:
            if xcond is not None:
                x_t_ref_cond = torch.hstack((x_t, xref, xcond))
            else:
                x_t_ref_cond = torch.hstack((x_t, xref))
            diff = self.vecfield(t, x_t_ref_cond) - u_t
        return self.manifold.inner(x_t, diff, diff).mean() / self.output_dim

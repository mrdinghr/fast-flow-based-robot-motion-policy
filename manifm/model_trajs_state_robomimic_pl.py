"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import torch

from torch.func import vjp, jvp, vmap, jacrev

from manifm.ema import EMA
from manifm.model.arch import tMLP, ProjectToTangent, Unbatch

from manifm.manifolds import geodesic
from manifm.solvers import projx_integrator
from manifm.model_pl import ManifoldFMLitModule
from manifm.model.uNet import Unet
from torchcfm.optimal_transport import OTPlanSampler


'''
Robomimic task with state based observation

functions
vecfield: learned vector field
sample_all: generate action series from observation condition vector
unpack_predictions_reference_conditioning_samples: get observation condition vector, prior samples and target samples 
                                                    from dataset
rfm_loss_fn: loss function
'''


class ManifoldFMRobomimicLitModule(ManifoldFMLitModule):
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

        # different dimension of state observation for different task
        if cfg.task == 'can':
            self.ref_object_feature = 14
        elif cfg.task == 'lift':
            self.ref_object_feature = 10
        elif cfg.task == 'square':
            self.ref_object_feature = 14
        elif cfg.task == 'tool_hang':
            self.ref_object_feature = 44
        else:
            assert False, 'Wrong Task setting'
        self.feature_dim = self.ref_object_feature + 3 + 4 + 2
        self.output_dim = self.dim * self.n_pred
        self.input_dim = self.output_dim + \
                         self.n_ref * self.feature_dim + \
                         self.n_cond * self.feature_dim + add_dim  # n x x_t (n x 3) + 1x ref (3) + 1x cond (3+1)

        self.model_type = cfg.model_type
        # Redefined the model of the vector field.
        if cfg.model_type == 'Unet':
            self.model = EMA(
                Unbatch(  # Ensures vmap works.
                    ProjectToTangent(  # Ensures we can just use Euclidean divergence.
                        Unet(input_dim=self.dim,
                             global_cond_dim=self.n_ref * self.feature_dim + \
                                             self.n_cond * self.feature_dim + add_dim,
                             down_dims=[256, 512, 1024]),
                        manifold=self.manifold,
                        metric_normalize=self.cfg.model.get("metric_normalize", False),
                        dim=self.output_dim  # As the last dims of x are the conditioning variable
                    )
                ),
                cfg.optim.ema_decay,
            )
        else:
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

        self.cfg = cfg
        self.small_var = False
        self.optimal_transport = cfg.optimal_transport
        print('fm manifm folder model')
        if self.optimal_transport:
            print('***********optimal transport***********')
            self.ot_sampler = OTPlanSampler(method='exact')

    @property
    def vecfield(self):
        return self.model

    @torch.no_grad()
    def sample_all(self, n_samples, device, xref, x0=None, different_sample=True, ode_steps=10):
        if x0 is None:
            # Sample from base distribution.
            x0 = (
                self.manifold.random_base(n_samples, self.output_dim, different_sample=different_sample)
                .reshape(n_samples, self.output_dim)
                .to(device)
            )
        if self.small_var:
            x0 *= 0.05
        if self.model_type == 'Unet':
            self.model.model.vecfield.vecfield.unet.global_cond = xref
            xs, _ = projx_integrator(
                self.manifold,
                self.vecfield,
                x0,
                t=torch.linspace(0, 1, ode_steps + 1).to(device),
                method="euler",
                projx=True,
                pbar=False,
            )
        else:
            # Wrapper for conditioning
            wrapper_vecfield = self.wrapper_red_cond(self.vecfield, xref, None)

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

    def unpack_predictions_reference_conditioning_samples(self, batch: torch.Tensor):
        B = batch['actions'].shape[0]
        x1 = batch['actions'][:, self.cfg.n_ref - 1:, :].reshape((B, 7 * self.n_pred)).to(self.device) # shape B * n_pred * 7
        xref_object = batch['obs']['object'][:, :self.cfg.n_ref, :]  # shape B * n_ref * 14
        xref_robo_pos = batch['obs']['robot0_eef_pos'][:, :self.cfg.n_ref, :]
        xref_robo_quat = batch['obs']['robot0_eef_quat'][:, :self.cfg.n_ref, :]
        xref_robo_grip = batch['obs']['robot0_gripper_qpos'][:, :self.cfg.n_ref, :]
        xref = torch.cat((xref_object, xref_robo_pos, xref_robo_quat, xref_robo_grip), dim=-1).reshape(
            (B, self.n_ref * self.feature_dim)).float().to(self.device)
        x0 = self.manifold.random_base(x1.shape[0], self.output_dim).to(x1)
        return x1, xref, x0

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
        x1, xref, x0 = self.unpack_predictions_reference_conditioning_samples(batch)
        if self.optimal_transport:
            ot_x0, ot_x1, index_list = self.ot_sampler.sample_plan(x0, x1, replace=False)
            ot_xref = xref[index_list]
            x0, x1, xref = ot_x0, ot_x1, ot_xref

        N = x1.shape[0]

        t = torch.rand(N).reshape(-1, 1).to(x1)

        # conditional vector field as Rectified flow on euclidean space
        x_t = x1 + (1 - t) * (x0 - x1)
        u_t = x1 - x0

        if self.model_type == 'Unet':
            self.model.model.vecfield.vecfield.unet.global_cond = xref
            diff = self.vecfield(t, x_t) - u_t
        else:
            assert False, 'model not implemented yet'
        return self.manifold.inner(x_t, diff, diff).mean() / self.output_dim

"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import torch

from torch.func import vjp, jvp, vmap, jacrev

from manifm.ema import EMA
from manifm.model.arch import tMLP, ProjectToTangent, Unbatch

from stable_manifolds_learntau import geodesic, projx_integrator, new_geodesic
from manifm.vision.resnet_models import get_resnet, replace_bn_with_gn
from stable_unet import StableUnetLearnTau, StableUnetLearnTauNew, StableUnetLearnTauStepEncoder
from stable_model_trajs_pl_learntau import SRFMTrajsModuleLearnTau
from torchcfm.optimal_transport import OTPlanSampler


'''
SRFMP model for euclidean and sphere PsuhT 

vecfield: learned vector field
sample_all: generate action series from observation condition vector xref
unpack_predictions_reference_conditioning_samples: get xref, prior sample x0, target sample x1 from training dataset
'''


class SRFMVisionResnetTrajsModuleLearnTau(SRFMTrajsModuleLearnTau):
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
        if self.cfg.data == 'pusht_vision_ref_cond' or self.cfg.data == 'pusht_vision_ref_cond_band':
            self.vision_feature_dim = 512 + 2  # ResNet18 output 1536; 4 pos dim
        elif self.cfg.data == 'pusht_sphere_vision_ref_cond' or self.cfg.data == 'pusht_sphere_vision_ref_cond_euc':
            self.vision_feature_dim = 512 + self.dim
        else:
            self.vision_feature_dim = 512  # ResNet18 has output dim of 512
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
                    StableUnetLearnTauStepEncoder(input_dim=self.dim,
                         global_cond_dim=self.n_ref * self.vision_feature_dim + \
                                         self.n_cond * self.vision_feature_dim + add_dim,
                         down_dims=[256, 512, 1024], output_dim=self.output_dim),
                    manifold=self.z_manifold,
                    metric_normalize=self.cfg.model.get("metric_normalize", False),
                    dim=self.output_dim + 1  # As the last dims of x are the conditioning variable
                )
            )
        elif cfg.model_type == 'tMLP':
            self.model = Unbatch(  # Ensures vmap works.
                ProjectToTangent(  # Ensures we can just use Euclidean divergence.
                    tMLP(  # Vector field in the ambient space.
                        self.input_dim,
                        d_out=self.output_dim,
                        d_model=cfg.model.d_model,
                        num_layers=cfg.model.num_layers,
                        actfn=cfg.model.actfn,
                        fourier=cfg.model.get("fourier", None),
                    ),
                    manifold=self.z_manifold,
                    metric_normalize=self.cfg.model.get("metric_normalize", False),
                    dim=self.output_dim + 1  # As the last dims of x are the conditioning variable
                )
            )
        else:
            assert False, 'wrong model type'

        self.cfg = cfg
        self.small_var = False
        self.optimal_transport = cfg.optimal_transport
        if self.optimal_transport:
            self.ot_sampler = OTPlanSampler(method='exact')
        print("number of parameters of VF model: {:e}".format(
            sum(p.numel() for p in self.model.parameters()))
        )

        print("number of parameters of Resnet model: {:e}".format(
            sum(p.numel() for p in self.vision_encoder.parameters()))
        )
        self.nets = EMA(torch.nn.ModuleDict({'vision_encoder': self.vision_encoder, 'vf_net': self.model}),
                        cfg.optim.ema_decay, )

    @property
    def vecfield(self):
        return self.model

    @torch.no_grad()
    def sample_all(self, n_samples, device, xref, xcond, z0=None, different_sample=True, adp=False, ode_steps=10):
        if z0 is None:
            # Sample from base distribution.
            x0 = (
                self.manifold.random_base(n_samples, self.output_dim, different_sample=different_sample)
                .reshape(n_samples, self.output_dim)
                .to(device)
            )
            tau0 = torch.zeros((n_samples, 1)).to(device)
            z0 = torch.hstack([x0, tau0])
        if self.small_var:
            print('small var')
            z0[..., :-1] *= 0.05
        if self.model_type == 'Unet':
            self.model.vecfield.vecfield.unet.global_cond = torch.hstack((xref, xcond))
            zs, _ = projx_integrator(
                manifold=self.z_manifold,
                odefunc=self.vecfield,
                z0=z0,
                lambda_tau=self.lambda_tau,
                ode_steps=ode_steps,
                method="euler",
                projx=True,
                pbar=False,
                adap_step=adp,
                local_coords=True,
            )
            # midpoint euler
        elif self.model_type == 'tMLP':
            # Wrapper for conditioning
            wrapper_vecfield = self.wrapper_red_cond(self.vecfield, xref, xcond)

            # Solve ODE.
            zs, _ = projx_integrator(
                self.z_manifold,
                wrapper_vecfield,
                z0,
                ode_steps=10,
                method="euler",
                projx=True,
                pbar=False,
            )
        else:
            assert False, 'model type not right'
        return zs

    def unpack_predictions_reference_conditioning_samples(self, batch: torch.Tensor):
        # Unpack batch vector into different components
        # Assumes greyscale images
        if (self.cfg.data == 'pusht_vision_ref_cond' or self.cfg.data == 'pusht_sphere_vision_ref_cond' or
                self.cfg.data == 'pusht_vision_ref_cond_band' or self.cfg.data == 'pusht_sphere_vision_ref_cond_euc'):
            B = batch['action'].shape[0]
            x1 = batch['action'].reshape((B, self.n_pred * self.dim))
            x0 = self.manifold.random_base(x1.shape[0], self.output_dim).to(x1)
            xref_pos = batch['obs']['agent_pos']
            xref_pos = xref_pos.reshape((B, (self.n_ref + self.n_cond) * self.dim)).to(self.device)
            xref_image = batch['obs']['image']
            xref_image_scalar = self.image_to_features(xref_image)
            xref = torch.hstack([xref_image_scalar, xref_pos])
            if self.small_var:
                x0 *= 0.05
            return x1, xref, None, x0
        else:
            if isinstance(batch, dict):
                x0 = batch["x0"]
                x1 = batch["x1"]
                xref = batch["xref"]
                xcond = batch["xcond"]
            else:
                x1 = batch[..., :self.output_dim]
                xref = batch[..., self.output_dim:self.output_dim + self.n_ref * self.image_dim ** 2]
                xcond = batch[..., self.output_dim + self.n_ref * self.image_dim ** 2:self.output_dim + (
                        self.n_ref + self.n_cond) * self.image_dim ** 2 + 1]
                x0 = self.manifold.random_base(x1.shape[0], self.output_dim).to(x1)

            # Transform images into feature vectors
            xcond_scalar = xcond[..., -1][:, None]
            xref_feature = self.image_to_features(xref)
            xcond_feature = self.image_to_features(xcond[..., :-1])
            xcond_feature_scalar = torch.hstack((xcond_feature, xcond_scalar))

        return x1, xref_feature, xcond_feature_scalar, x0

    def image_to_features(self, x):
        # x_image = x.reshape((-1, self.image_dim, self.image_dim)).unsqueeze(-3).repeat(1, 3, 1, 1)
        # return self.vision_encoder(x_image)
        if self.cfg.data == 'pusht_vision_ref_cond' or self.cfg.data == 'pusht_sphere_vision_ref_cond' or self.cfg.data == 'pusht_vision_ref_cond_band' or self.cfg.data == 'pusht_sphere_vision_ref_cond_euc':
            image_feaeture = self.vision_encoder(x.flatten(end_dim=1).float().to(self.device))
            image_feaeture = image_feaeture.reshape(*x.shape[:2], -1)
            return image_feaeture.flatten(start_dim=1)
        else:
            batch = x.shape[0]
            x_image = x.reshape((-1, self.image_dim, self.image_dim)).unsqueeze(-3).repeat(1, 3, 1, 1)
            return self.vision_encoder(x_image).reshape(batch, -1)

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

        if self.optimal_transport:
            x0, x1 = self.ot_sampler.sample_plan(x0, x1)

        tau0 = torch.zeros(1).to(x0)
        tau1 = torch.ones(1).to(x0)
        N = x1.shape[0]
        tau = torch.rand(N).reshape(-1, 1).to(x1)
        if torch.any(tau == 1):
            one_id, _ = torch.where(tau == 1)
            tau[one_id] -= 0.02
        # z0 = torch.hstack([x0, tau])
        # z1 = torch.hstack([x1, tau1.repeat(N)])
        t = -torch.log((tau - tau1) / (tau0 - tau1)) / self.lambda_tau

        def cond_u(x0, x1, t):
            path = geodesic(self.manifold, x0, x1, self.lambda_x)
            x_t, ux_t = jvp(path, (t,), (torch.ones_like(t).to(t),))
            return x_t, ux_t

        # here ux_t is the derivative of x flow to tau
        x_t, ux_t = vmap(cond_u)(x0, x1, t)
        x_t = x_t.reshape(N, self.output_dim)
        ux_t = ux_t.reshape(N, self.output_dim)
        tau_t = tau
        utau_t = - self.lambda_tau * (tau_t - tau1)
        z_t = torch.hstack([x_t, tau_t])
        uz_t = torch.hstack([ux_t, utau_t])

        self.check_automatic_dt(uz_t=uz_t, x1=x1, zt=z_t)

        if self.model_type == 'Unet':
            if xcond is not None:
                self.model.vecfield.vecfield.unet.global_cond = torch.hstack((xref, xcond))
            else:
                self.model.vecfield.vecfield.unet.global_cond = xref
            diff = self.vecfield(tau, z_t) - uz_t
            # diff *= 3 * torch.exp(tau)
        else:
            if xcond is not None:
                x_t_ref_cond = torch.hstack((z_t, xref, xcond))
            else:
                x_t_ref_cond = torch.hstack((z_t, xref))
            diff = self.vecfield(tau, x_t_ref_cond) - uz_t
        return self.z_manifold.inner(z_t, diff, diff).mean() / self.output_dim

    def check_automatic_dt(self, zt, x1, uz_t):
        z1 = torch.hstack([x1, torch.ones((x1.shape[0], 1)).to(self.device)])
        gen_z1 = self.z_manifold.expmap(zt, uz_t / self.lambda_x)
        gen_z1 = self.z_manifold.projx(gen_z1)
        print(z1 - gen_z1)

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

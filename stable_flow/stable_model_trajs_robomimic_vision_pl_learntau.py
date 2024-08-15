"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import torch

from torch.func import vjp, jvp, vmap, jacrev
from typing import Any, List

from manifm.ema import EMA
from manifm.model.arch import tMLP, ProjectToTangent, Unbatch

from stable_manifolds_learntau import geodesic, projx_integrator
from stable_unet import StableUnetLearnTauStepEncoder
from stable_model_trajs_pl_learntau import SRFMTrajsModuleLearnTau
from manifm.vision.resnet_models import get_resnet, replace_bn_with_gn
import torchvision.transforms as Transform


class SRFMRobomimicVisionLTModule(SRFMTrajsModuleLearnTau):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.n_pred = cfg.n_pred
        self.n_ref = cfg.n_ref
        self.n_cond = cfg.n_cond
        self.w_cond = cfg.w_cond
        if self.n_cond > 0:
            add_dim = 0
        else:
            add_dim = 0

        self.img_feature_dim = 512
        self.feature_dim = self.img_feature_dim * 2 + 3 + 4 + 2
        self.output_dim = self.dim * self.n_pred
        self.input_dim = self.output_dim + \
                         self.n_ref * self.feature_dim + \
                         self.n_cond * self.feature_dim + add_dim

        self.model_type = cfg.model_type
        # Redefined the model of the vector field.
        if cfg.model_type == 'Unet':
            self.model = Unbatch(  # Ensures vmap works.
                    ProjectToTangent(  # Ensures we can just use Euclidean divergence.
                        StableUnetLearnTauStepEncoder(input_dim=self.dim,
                             global_cond_dim=self.n_ref * self.feature_dim + \
                                             self.n_cond * self.feature_dim + add_dim,
                             down_dims=[256, 512, 1024], output_dim=self.output_dim),
                        manifold=self.z_manifold,
                        metric_normalize=self.cfg.model.get("metric_normalize", False),
                        dim=self.output_dim + 1  # As the last dims of x are the conditioning variable
                    )
                )
        else:
            assert False, 'network type wrongly set'

        self.vision_encoder = get_resnet('resnet18')
        self.vision_encoder = replace_bn_with_gn(self.vision_encoder)

        self.grip_vision_encoder = get_resnet('resnet18')
        self.grip_vision_encoder = replace_bn_with_gn(self.grip_vision_encoder)
        self.nets = EMA(torch.nn.ModuleDict({'vision_encoder': self.vision_encoder, 'vf_net': self.model,
                                             'grip_vision_encoder': self.grip_vision_encoder}), cfg.optim.ema_decay)

        self.cfg = cfg
        self.small_var = False
        if cfg.task == 'tool_hang':
            self.crop_transform = Transform.RandomCrop((216, 216))
        else:
            self.crop_transform = Transform.RandomCrop((76, 76))

    @property
    def vecfield(self):
        return self.model

    @torch.no_grad()
    def sample_all(self, n_samples, device, xref, xcond=None, z0=None, different_sample=True, adp=False, ode_steps=10):
        if z0 is None:
            # Sample from base distribution.
            sample = self.manifold.random_base(n_samples, self.output_dim, different_sample=different_sample)
            sample = sample.reshape((n_samples, self.cfg.n_pred, self.dim))
            if self.small_var:
                sample[:, :, 0:3] *= 0.05
                sample[:, :, -1] *= 0.05
                sample = sample.reshape((n_samples, self.cfg.n_pred * self.dim))
            x0 = (sample.reshape((n_samples, self.output_dim)).to(device))
            tau0 = torch.zeros((n_samples, 1)).to(self.device)
            z0 = torch.hstack([x0, tau0])
        if self.model_type == 'Unet':
            self.model.vecfield.vecfield.unet.global_cond = xref
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
                local_coords=False
            )
            # midpoint euler
        elif self.model_type == 'tMLP':
            assert False, 'model type not right'
            # Wrapper for conditioning
            wrapper_vecfield = self.wrapper_red_cond(self.vecfield, xref, None)

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
        B = batch['actions'].shape[0]
        x1 = batch['actions'][:, self.cfg.n_ref - 1:, :].reshape((B, 7 * self.n_pred)).to(
            self.device)  # shape B * n_pred * 7
        xref_image = batch['obs']['agentview_image'][:, :self.cfg.n_ref, :]
        xref_image = xref_image.moveaxis(-1, -3).float()
        xref_image_scalar = self.image_to_features(xref_image)  # shape B * n_ref * 14
        xref_robo_pos = batch['obs']['robot0_eef_pos'][:, :self.cfg.n_ref, :]
        xref_robo_quat = batch['obs']['robot0_eef_quat'][:, :self.cfg.n_ref, :]
        xref_robo_grip = batch['obs']['robot0_gripper_qpos'][:, :self.cfg.n_ref, :]
        xref_grip_image = batch['obs']['robot0_eye_in_hand_image'][:, :self.cfg.n_ref, :]
        xref_grip_image = xref_grip_image.moveaxis(-1, -3).float()
        xref_grip_iamge_scalar = self.grip_image_to_features(xref_grip_image)
        xref = torch.cat((xref_image_scalar, xref_grip_iamge_scalar, xref_robo_pos, xref_robo_quat, xref_robo_grip),
                         dim=-1).reshape(
            (B, self.n_ref * self.feature_dim)).float().to(self.device)
        x0 = self.manifold.random_base(x1.shape[0], self.output_dim).to(x1)
        return x1, xref, x0

    def image_to_features(self, x):
        x = self.crop_transform(x)
        image_feaeture = self.vision_encoder(x.flatten(end_dim=1))
        return image_feaeture.reshape(*x.shape[:2], -1)

    def grip_image_to_features(self, x):
        x = self.crop_transform(x)
        image_feaeture = self.grip_vision_encoder(x.flatten(end_dim=1))
        return image_feaeture.reshape(*x.shape[:2], -1)

    class wrapper_red_cond(torch.nn.Module):
        """Wraps model to torchdyn compatible format."""

        def __init__(self, vecfield, ref, cond):
            super().__init__()
            self.vecfield = vecfield
            self.ref = ref
            self.cond = cond

        def forward(self, t, x):
            return self.vecfield(t, torch.hstack((x, self.ref, self.cond)))

    def rfm_loss_fn(self, batch: torch.Tensor, train=True):
        x1, xref, x0 = self.unpack_predictions_reference_conditioning_samples(batch)

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

        if self.model_type == 'Unet':
            self.model.vecfield.vecfield.unet.global_cond = xref
            diff = self.vecfield(tau, z_t) - uz_t
            # diff *= 3 * torch.exp(tau)
        else:
            x_t_ref_cond = torch.hstack((z_t, xref))
            diff = self.vecfield(tau, x_t_ref_cond) - uz_t
        if not train:
            diff[..., -1] *= 0.0  # assumption that tau vf overfit
        return self.z_manifold.inner(z_t, diff, diff).mean() / self.output_dim

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

    def loss_fn(self, batch: torch.Tensor, train=True):
        return self.rfm_loss_fn(batch, train=train)

    def validation_step(self, batch: Any, batch_idx: int):
        loss = self.loss_fn(batch, train=False)

        if torch.isfinite(loss):
            # log train metrics
            self.log("val/loss", loss, on_step=True, on_epoch=True)
            self.val_metric.update(loss)
        else:
            # skip step if loss is NaN.
            print(f"Skipping iteration because loss is {loss.item()}.")
            return None
        return {"loss": loss}

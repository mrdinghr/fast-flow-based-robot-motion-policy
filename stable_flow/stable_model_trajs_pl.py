"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from typing import Any, List

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.func import vjp, jvp, vmap, jacrev

from manifm.ema import EMA
from manifm.model.arch import tMLP, ProjectToTangent, Unbatch

from stable_unet import StableUnet
from manifm.datasets import get_manifold
from stable_manifolds import geodesic, projx_integrator
from torchmetrics import MeanMetric, MinMetric


class SRFMTrajsModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.train_metric = MeanMetric()
        self.val_metric = MeanMetric()
        self.val_metric_best = MinMetric()

        self.cfg = cfg
        self.manifold, self.dim = get_manifold(cfg)
        self.n_pred = cfg.n_pred
        self.n_ref = cfg.n_ref
        self.n_cond = cfg.n_cond
        self.w_cond = cfg.w_cond

        self.lambda_x = cfg.lambda_x
        self.lambda_tau = cfg.lambda_tau

        if "obs_dim" in cfg:
            self.obs_dim = cfg.obs_dim
        else:
            self.obs_dim = self.dim
        if self.n_cond > 0:
            add_dim = 1
        else:
            add_dim = 0
        self.output_dim = self.dim * self.n_pred
        self.input_dim = self.output_dim + self.n_ref * self.obs_dim + self.n_cond * self.obs_dim + add_dim  # n x x_t (n x 3) + 1x ref (3) + 1x cond (3+1)

        self.model_type = cfg.model_type
        # Redefined the model of the vector field.
        if cfg.model_type == 'Unet':
            self.model = EMA(
                Unbatch(  # Ensures vmap works.
                    ProjectToTangent(  # Ensures we can just use Euclidean divergence.
                        StableUnet(input_dim=self.dim,
                             global_cond_dim=self.n_ref * self.obs_dim + self.n_cond * self.obs_dim + add_dim),
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
        self.previous_x0 = None
        self.small_var = False
        self.diff_prior = True

    @property
    def vecfield(self):
        return self.model

    @torch.no_grad()
    def sample_all(self, n_samples, device, xref, xcond, x0=None, shift_mean=False):
        if x0 is None:
            # Sample from base distribution.
            # x0 = (
            #     self.manifold.random_base(n_samples, self.output_dim, different_sample=different_sample)
            #     .reshape(n_samples, self.output_dim)
            #     .to(device)
            # )
            x0 = (
                self.manifold.random_base(n_samples, self.output_dim, different_sample=self.diff_prior)
                .reshape(n_samples, self.output_dim)
                .to(device)
            )
            if self.small_var:
                print('small var')
                x0 *= 0.05
            print(x0[0])
            # if self.previous_x0 is not None and shift_mean:
            #     x0 = self.previous_x0 + x0 * 0.05
            #     print(x0[0])
            #     print(self.previous_x0[0])
            if shift_mean:
                self.previous_x0 = x0

        if self.model_type == 'Unet':
            self.model.model.vecfield.vecfield.unet.global_cond = torch.hstack((xref, xcond))
            xs, _ = projx_integrator(
                manifold=self.manifold,
                odefunc=self.vecfield,
                x0=x0,
                lambda_tau=self.lambda_tau,
                ode_steps=10,
                method="euler",
                projx=True,
                pbar=True,
            )
        # Wrapper for conditioning
        elif self.model_type == 'tMLP':
            wrapper_vecfield = self.wrapper_red_cond(self.vecfield, xref, xcond)
            xs, _ = projx_integrator(
                manifold=self.manifold,
                odefunc=wrapper_vecfield,
                x0=x0,
                lambda_tau=self.lambda_tau,
                ode_steps=1000,
                method="euler",
                projx=True,
                pbar=True,
            )
        else:
            assert False, 'model type not right'
        return xs

    def unpack_predictions_reference_conditioning_samples(self, batch: torch.Tensor):
        if isinstance(batch, dict):
            # This is the case for some dataset, e.g., Pusht
            action = batch["action"]
            obs = batch["obs"]
            # xcond = batch["xcond"]
            batch_size = action.shape[0]
            x1 = action.reshape([batch_size, -1])
            xref = obs.reshape([batch_size, -1])
            xcond = torch.zeros(batch_size, 0, dtype=x1.dtype, device=x1.device)
            x0 = self.manifold.random_base(x1.shape[0], self.output_dim, different_sample=self.diff_prior).to(x1)
        else:
            x1 = batch[..., :self.dim * self.n_pred]
            xref = batch[..., self.dim * self.n_pred:self.dim * self.n_pred + self.obs_dim * self.n_ref]
            xcond = batch[...,
                    self.dim * self.n_pred + self.obs_dim * self.n_ref:self.dim * self.n_pred + self.obs_dim * (
                            self.n_ref + self.n_cond) + 1]
            x0 = self.manifold.random_base(x1.shape[0], self.output_dim, different_sample=self.diff_prior).to(x1)

        return x1, xref, xcond, x0

    class wrapper_red_cond(torch.nn.Module):
        """Wraps model to torchdyn compatible format."""

        def __init__(self, vecfield, ref, cond, model_type='tMLP'):
            super().__init__()
            self.vecfield = vecfield
            self.ref = ref
            self.cond = cond
            self.model_type = model_type

        def forward(self, t, x):
            if self.model_type == 'Unet':
                self.vecfield.model.vecfield.vecfield.unet.global_cond = torch.hstack((self.ref, self.cond))
                return self.vecfield(t, x)
            else:
                return self.vecfield(t, torch.hstack((x, self.ref, self.cond)))

    def rfm_loss_fn(self, batch: torch.Tensor):
        x1, xref, xcond, x0 = self.unpack_predictions_reference_conditioning_samples(batch)

        tau0 = torch.zeros(1).to(x0)
        tau1 = torch.ones(1).to(x0)
        N = x1.shape[0]
        tau = torch.rand(N).reshape(-1, 1).to(x1)
        if torch.any(tau == 1):
            one_id, _ = torch.where(tau == 1)
            tau[one_id] -= 0.02
        t = -torch.log((tau - tau1) / (tau0 - tau1)) / self.lambda_tau

        def cond_u(x0, x1, t):
            path = geodesic(self.manifold, x0, x1, self.lambda_x)
            x_t, u_t = jvp(path, (t,), (torch.ones_like(t).to(t),))
            return x_t, u_t

        # here ux_t is the derivative of x flow to tau
        x_t, ux_t = vmap(cond_u)(x0, x1, t)
        x_t = x_t.reshape(N, self.output_dim)
        ux_t = ux_t.reshape(N, self.output_dim)

        if self.model_type == 'Unet':
            self.model.model.vecfield.vecfield.unet.global_cond = torch.hstack((xref, xcond))
            diff = self.vecfield(tau, x_t) - ux_t
        else:
            x_t_ref_cond = torch.hstack((x_t, xref, xcond))
            diff = self.vecfield(tau, x_t_ref_cond) - ux_t
        return self.manifold.inner(x_t, diff, diff).mean() / self.output_dim

    def loss_fn(self, batch: torch.Tensor):
        return self.rfm_loss_fn(batch)

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.loss_fn(batch)

        if torch.isfinite(loss):
            # log train metrics
            self.log("train/loss", loss, on_step=True, on_epoch=True)
            self.train_metric.update(loss)
        else:
            # skip step if loss is NaN.
            print(f"Skipping iteration because loss is {loss.item()}.")
            return None

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        self.train_metric.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        loss = self.loss_fn(batch)

        if torch.isfinite(loss):
            # log train metrics
            self.log("val/loss", loss, on_step=True, on_epoch=True)
            self.val_metric.update(loss)
        else:
            # skip step if loss is NaN.
            print(f"Skipping iteration because loss is {loss.item()}.")
            return None

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        val_loss = self.val_metric.compute()  # get val accuracy from current epoch
        self.val_metric_best.update(val_loss)
        self.log(
            "val/loss_best",
            self.val_metric_best.compute(),
            on_epoch=True,
            prog_bar=True,
        )
        self.val_metric.reset()
        print('validation end')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
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
        if isinstance(self.model, EMA):
            self.model.update_ema()

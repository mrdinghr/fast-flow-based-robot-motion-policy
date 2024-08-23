"""Copyright (c) Meta Platforms, Inc. and affiliates."""
import time

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
from manifm.solvers import projx_integrator_return_last, projx_integrator
from manifm.model_pl import div_fn, output_and_div, ManifoldFMLitModule
from manifm.vision.resnet_models import get_resnet, replace_bn_with_gn
from manifm.model.uNet import Unet


'''
class for RFMP PickPlace real robot tasks
without vision encoder in training backbone
Note: this is not used anymore
'''


class ManifoldVisionTrajectoriesDishGraspFMLitModule(ManifoldFMLitModule):
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

        # Vision encoder for over-the-shoulder camera
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
            self.model = EMA(
                Unbatch(  # Ensures vmap works.
                    ProjectToTangent(  # Ensures we can just use Euclidean divergence.
                        Unet(input_dim=self.dim,
                             global_cond_dim=self.n_ref * self.vision_feature_dim + \
                                             self.n_cond * self.vision_feature_dim + add_dim,
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
        self.use_gripper_img = cfg.use_gripper_img

    @property
    def vecfield(self):
        return self.model

    @torch.no_grad()
    def compute_cost(self, batch):
        if isinstance(batch, dict):
            x0 = batch["x0"]
        else:
            x0 = (
                self.manifold.random_base(batch.shape[0], self.output_dim)
                .reshape(batch.shape[0], self.output_dim)
                .to(batch.device)
            )

        x1, xref, xcond, _ = self.unpack_predictions_reference_conditioning_samples(batch)

        # Wrapper for conditioning
        wrapper_vecfield = self.wrapper_red_cond(self.vecfield, xref, xcond)

        # Solve ODE.
        x1 = odeint(
            wrapper_vecfield,
            x0,
            t=torch.linspace(0, 1, 2).to(x0.device),
            atol=self.cfg.model.atol,
            rtol=self.cfg.model.rtol,
        )[-1]

        x1 = self.manifold.projx(x1)

        return self.manifold.dist(x0, x1)

    @torch.no_grad()
    def sample(self, n_samples, device, xref, xcond, x0=None):
        if x0 is None:
            # Sample from base distribution.
            x0 = (
                self.manifold.random_base(n_samples, self.output_dim)
                .reshape(n_samples, self.output_dim)
                .to(device)
            )

        local_coords = self.cfg.get("local_coords", False)
        eval_projx = self.cfg.get("eval_projx", False)

        # Wrapper for conditioning
        wrapper_vecfield = self.wrapper_red_cond(self.vecfield, xref, xcond)

        # Solve ODE.
        if not eval_projx and not local_coords:
            # If no projection, use adaptive step solver.
            x1 = odeint(
                wrapper_vecfield,
                x0,
                t=torch.linspace(0, 1, 2).to(device),
                atol=self.cfg.model.atol,
                rtol=self.cfg.model.rtol,
                options={"min_step": 1e-5}
            )[-1]
        else:
            # If projection, use 1000 steps.
            x1 = projx_integrator_return_last(
                self.manifold,
                wrapper_vecfield,
                x0,
                t=torch.linspace(0, 1, 1001).to(device),
                method="euler",
                projx=eval_projx,
                local_coords=local_coords,
                pbar=True,
            )
        # x1 = self.manifold.projx(x1)
        return x1

    @torch.no_grad()
    def sample_all(self, n_samples, device, xref, xcond=None, x0=None, different_sample=False, ode_steps=11):
        if self.small_var and x0 is not None:
            x0 = x0.reshape((n_samples, self.cfg.n_pred, self.dim))
            x0[:, :, 0:3] *= 0.05
            x0[:, :, -1] *= 0.05
            x0 = x0.reshape((n_samples, self.cfg.n_pred * self.dim))
        if x0 is None:
            # Sample from base distribution.
            cur_time = time.time()
            sample = self.manifold.random_base(n_samples, self.output_dim, different_sample=different_sample)
            sample = sample.reshape((n_samples, self.cfg.n_pred, self.dim))
            if self.small_var:
                sample[:, :, 0:3] *= 0.05
                sample[:, :, -1] *= 0.05
                sample = sample.reshape((n_samples, self.cfg.n_pred * self.dim))
            print('sample x0 use time ', time.time() - cur_time)
            x0 = (sample.reshape((n_samples, self.output_dim)).to(device))

        if self.model_type == 'Unet':
            if xcond is not None:
                self.model.model.vecfield.vecfield.unet.global_cond = torch.hstack((xref, xcond))
            else:
                self.model.model.vecfield.vecfield.unet.global_cond = xref
            xs, _ = projx_integrator(
                self.manifold,
                self.vecfield,
                x0,
                t=torch.linspace(0, 1, ode_steps).to(device),
                method="euler",
                projx=True,
                pbar=False,
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

    @torch.no_grad()
    def compute_exact_loglikelihood(
            self,
            batch: torch.Tensor,
            t1: float = 1.0,
            return_projx_error: bool = False,
            num_steps=1000,
    ):
        """Computes the negative log-likelihood of a batch of data."""

        # try:
        nfe = [0]

        div_mode = self.cfg.get("div_mode", "exact")

        with torch.inference_mode(mode=False):
            xbatch, xref, xcond, _ = self.unpack_predictions_reference_conditioning_samples(batch)
            wrapper_vecfield = self.wrapper_red_cond(self.vecfield, xref, xcond)

            v = None
            if div_mode == "rademacher":
                v = torch.randint(low=0, high=2, size=batch.shape).to(batch) * 2 - 1
                v = v[..., :self.output_dim]

            def odefunc(t, tensor):
                nfe[0] += 1
                t = t.to(tensor)
                x = tensor[..., : self.output_dim]

                vecfield = lambda x: wrapper_vecfield(t, x)
                dx, div = output_and_div(vecfield, x, v=v, div_mode=div_mode)

                if hasattr(self.manifold, "logdetG"):
                    def _jvp(x, v):
                        return jvp(self.manifold.logdetG, (x,), (v,))[1]

                    corr = vmap(_jvp)(x, dx)
                    div = div + 0.5 * corr.to(div)

                div = div.reshape(-1, 1)
                del t, x
                return torch.cat([dx, div], dim=-1)

            # Solve ODE on the product manifold of data manifold x euclidean.
            product_man = ProductManifold(
                (self.manifold, self.output_dim), (Euclidean(), 1)
            )
            # state1 = torch.cat([batch, torch.zeros_like(batch[..., :1])], dim=-1)
            state1 = torch.cat([xbatch, torch.zeros_like(xbatch[..., :1])], dim=-1)

            local_coords = self.cfg.get("local_coords", False)
            eval_projx = self.cfg.get("eval_projx", False)

            with torch.no_grad():
                if not eval_projx and not local_coords:
                    # If no projection, use adaptive step solver.
                    state0 = odeint(
                        odefunc,
                        state1,
                        t=torch.linspace(t1, 0, 2).to(batch),
                        atol=self.cfg.model.atol,
                        rtol=self.cfg.model.rtol,
                        method="dopri5",
                        options={"min_step": 1e-5},
                    )[-1]
                else:
                    # If projection, use 1000 steps.
                    state0 = projx_integrator_return_last(
                        product_man,
                        odefunc,
                        state1,
                        t=torch.linspace(t1, 0, num_steps + 1).to(batch),
                        method="euler",
                        projx=eval_projx,
                        local_coords=local_coords,
                        pbar=True,
                    )

            # log number of function evaluations
            self.log("nfe", nfe[0], prog_bar=True, logger=True)

            x0, logdetjac = state0[..., : self.output_dim], state0[..., -1]
            x0_ = x0
            x0 = self.manifold.projx(x0)

            # log how close the final solution is to the manifold.
            integ_error = (x0[..., : self.output_dim] - x0_[..., : self.output_dim]).abs().max()
            self.log("integ_error", integ_error)

            logp0 = self.manifold.base_logprob(x0)
            logp1 = logp0 + logdetjac

            if self.cfg.get("normalize_loglik", False):
                logp1 = logp1 / self.output_dim

            # Mask out those that left the manifold
            masked_logp1 = logp1
            if isinstance(self.manifold, SPD):
                mask = integ_error < 1e-5
                self.log("frac_within_manifold", mask.sum() / mask.nelement())
                masked_logp1 = logp1[mask]

            if return_projx_error:
                return logp1, integ_error
            else:
                return masked_logp1
        # except:
        #     traceback.print_exc()
        #     return torch.zeros(batch.shape[0]).to(batch)

    def unpack_predictions_reference_conditioning_samples(self, batch: torch.Tensor):
        # Unpack batch vector into different components
        B = batch['observation']['v_fix'].shape[0]

        x1_pos_quat = batch['future_pos_quat']
        x1_gripper = batch['future_gripper']
        x1 = torch.cat((x1_pos_quat, x1_gripper), dim=2).reshape((B, self.n_pred * self.dim)).to(self.device)
        x0 = self.manifold.random_base(x1.shape[0], self.output_dim).to(x1)

        # image reference scalar: fix_img, gripper_img
        if self.use_gripper_img:
            fix_img = batch['observation']['v_fix'].to(self.device)  # 3 * H * W
            gripper_img = batch['observation']['v_gripper'].to(self.device)
            fix_img_feature = self.image_to_features(fix_img).reshape((B, self.n_ref, 512))  # B * (n_ref * 512)  #
            gripper_img_feature = self.image_to_features(gripper_img).reshape((B, self.n_ref, 512))
            img_scalar = torch.cat((fix_img_feature, gripper_img_feature), dim=2).reshape((B, self.n_ref * 2 * 512))  # shape B * 2048
        else:
            fix_img = batch['observation']['v_fix'].to(self.device)
            fix_img_feature = self.image_to_features(fix_img).reshape((B, self.n_ref, 512)).to(self.device)
            img_scalar = fix_img_feature.reshape((B, self.n_ref * 512))
        xref_pos_quat_pre = batch['previous_pos_quat'].squeeze(dim=1)
        xref_pos_quat_cur = batch['future_pos_quat'][:, 0, :]
        xref_pos_quat = torch.cat((xref_pos_quat_pre, xref_pos_quat_cur), dim=1).to(self.device)
        xref_gripper_pre = batch['previous_gripper'].squeeze(dim=1)
        xref_gripper_cur = batch['smooth_future_gripper'][:, 0, :]
        xref_gripper = torch.cat((xref_gripper_pre, xref_gripper_cur), dim=1).to(self.device)  #shape B * 2
        xref = torch.cat((img_scalar, xref_pos_quat, xref_gripper), dim=1)
        return x1, xref, None, x0

    def image_to_features(self, x):
        image_feaeture = self.vision_encoder(x.flatten(end_dim=1))
        image_feaeture = image_feaeture.reshape(*x.shape[:2], -1)
        return image_feaeture.flatten(start_dim=1)

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

        # set the observation and time condition vector
        if self.model_type == 'Unet':
            if xcond is not None:
                self.model.model.vecfield.vecfield.unet.global_cond = torch.hstack((xref, xcond))
            else:
                self.model.model.vecfield.vecfield.unet.global_cond = xref
            diff = self.vecfield(t, x_t) - u_t
        else:
            if xcond is not None:
                x_t_ref_cond = torch.hstack((x_t, xref, xcond))
            else:
                x_t_ref_cond = torch.hstack((x_t, xref))
            diff = self.vecfield(t, x_t_ref_cond) - u_t
        return self.manifold.inner(x_t, diff, diff).mean() / self.output_dim

"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import torch

from torch.func import vjp, jvp, vmap, jacrev

from manifm.ema import EMA
from manifm.model.arch import tMLP, ProjectToTangent, Unbatch

from manifm.manifolds import geodesic
from manifm.solvers import projx_integrator
from manifm.model_pl import ManifoldFMLitModule
from manifm.model.uNet import Unet
from manifm.vision.resnet_models import get_resnet, replace_bn_with_gn
import torchvision.transforms as Transform

'''
Robomimimc task with vision observation, 2 camera: over-the-shoulder camera and in-hand camera

functions
vecfield: learned vector field
sample_all: generate action series from observation condition vector
unpack_predictions_reference_conditioning_samples: get observation condition vector, prior samples and target samples 
                                                    from dataset
image_to_features: vision encoder of over-the-shoulder camera
grip_image_to_features: vision encoder of in-hand camera
rfm_loss_fn: loss function
configure_optimizers: configure optimizer
optimizer_step: update learned parameters
'''


class ManifoldFMImageRobomimicLitModule(ManifoldFMLitModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.n_pred = cfg.n_pred
        self.n_ref = cfg.n_ref
        self.n_cond = cfg.n_cond
        self.w_cond = cfg.w_cond
        self.task = cfg.task
        if self.n_cond > 0:
            add_dim = 1
        else:
            add_dim = 0

        self.img_feature_dim = 512
        # img_feature for 2 cameras, 3 for transition, 4 for quarternion, 2 for gripper
        self.feature_dim = self.img_feature_dim * 2 + 3 + 4 + 2
        self.output_dim = self.dim * self.n_pred
        self.input_dim = self.output_dim + \
                         self.n_ref * self.feature_dim + \
                         self.n_cond * self.feature_dim + add_dim  # n x x_t (n x 3) + 1x ref (3) + 1x cond (3+1)

        self.model_type = cfg.model_type
        # Redefined the model of the vector field.
        if cfg.model_type == 'Unet':
            self.model = Unbatch(  # Ensures vmap works.
                ProjectToTangent(  # Ensures we can just use Euclidean divergence.
                    Unet(input_dim=self.dim,
                         global_cond_dim=self.n_ref * self.feature_dim + \
                                         self.n_cond * self.feature_dim + add_dim,
                         down_dims=[256, 512, 1024]),
                    manifold=self.manifold,
                    metric_normalize=self.cfg.model.get("metric_normalize", False),
                    dim=self.output_dim  # As the last dims of x are the conditioning variable
                )
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

        # Vision Encoder for over-the-shoulder camera and in-hand camera
        self.vision_encoder = get_resnet('resnet18')
        self.vision_encoder = replace_bn_with_gn(self.vision_encoder)

        self.grip_vision_encoder = get_resnet('resnet18')
        self.grip_vision_encoder = replace_bn_with_gn(self.grip_vision_encoder)
        self.nets = EMA(torch.nn.ModuleDict({'vision_encoder': self.vision_encoder, 'vf_net': self.model,
                                             'grip_vision_encoder': self.grip_vision_encoder}))

        # check the parameters number of learned vector field and vision encoder
        print("number of parameters of VF model: {:e}".format(
            sum(p.numel() for p in self.model.parameters()))
        )

        print("number of parameters of Resnet model: {:e}".format(
            sum(p.numel() for p in self.vision_encoder.parameters()))
        )

        self.cfg = cfg
        self.small_var = False

        # duriing training random crop the image
        if cfg.task == 'tool_hang':
            self.crop_transform = Transform.RandomCrop((216, 216))
        else:
            self.crop_transform = Transform.RandomCrop((76, 76))

    @property
    def vecfield(self):
        return self.model

    @torch.no_grad()
    def sample_all(self, n_samples, device, xref, x0=None, different_sample=True, ode_steps=10):
        '''
        generate action series from observation condition vector xref

        n_samples: number of generated action series
        device: cuda, cpu
        xref: observation condition vector
        x0: prior samples
        different_sample: whether the prior sample is same at each dimension
        ode_steps: ODE solving steps

        return: generated action series
        '''
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
            self.model.vecfield.vecfield.unet.global_cond = xref
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
        x1 = batch['actions'][:, self.cfg.n_ref - 1:, :].reshape((B, 7 * self.n_pred)).to(
            self.device)  # shape B * n_pred * 7
        if self.task == 'tool_hang':
            xref_image = batch['obs']['sideview_image'][:, :self.cfg.n_ref, :]
        else:
            xref_image = batch['obs']['agentview_image'][:, :self.cfg.n_ref, :]
        xref_image = xref_image.moveaxis(-1, -3).float()
        xref_image_scalar = self.image_to_features(xref_image)  # shape B * n_ref * 14
        xref_robo_pos = batch['obs']['robot0_eef_pos'][:, :self.cfg.n_ref, :]
        xref_robo_quat = batch['obs']['robot0_eef_quat'][:, :self.cfg.n_ref, :]
        xref_robo_grip = batch['obs']['robot0_gripper_qpos'][:, :self.cfg.n_ref, :]
        xref_grip_image = batch['obs']['robot0_eye_in_hand_image'][:, :self.cfg.n_ref, :]
        xref_grip_image = xref_grip_image.moveaxis(-1, -3).float()
        xref_grip_iamge_scalar = self.grip_image_to_features(xref_grip_image)
        xref = torch.cat((xref_image_scalar, xref_grip_iamge_scalar, xref_robo_pos, xref_robo_quat, xref_robo_grip), dim=-1).reshape(
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

    def rfm_loss_fn(self, batch: torch.Tensor):
        x1, xref, x0 = self.unpack_predictions_reference_conditioning_samples(batch)

        N = x1.shape[0]

        t = torch.rand(N).reshape(-1, 1).to(x1)

        # conditional vector field as Rectified flow on Euclidean space
        x_t = x1 + (1 - t) * (x0 - x1)
        u_t = x1 - x0

        if self.model_type == 'Unet':
            self.model.vecfield.vecfield.unet.global_cond = xref
            diff = self.vecfield(t, x_t) - u_t
        else:
            assert False, 'model not implemented yet'
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

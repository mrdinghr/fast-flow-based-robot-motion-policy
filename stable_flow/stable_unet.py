# @markdown ### **Network**
# @markdown
# @markdown Defines a 1D UNet architecture `ConditionalUnet1D`
# @markdown as the noies prediction network
# @markdown
# @markdown Components
# @markdown - `SinusoidalPosEmb` Positional encoding for the diffusion iteration k
# @markdown - `Downsample1d` Strided convolution to reduce temporal resolution
# @markdown - `Upsample1d` Transposed convolution to increase temporal resolution
# @markdown - `Conv1dBlock` Conv1d --> GroupNorm --> Mish
# @markdown - `ConditionalResidualBlock1D` Takes two inputs `x` and `cond`. \
# @markdown `x` is passed through 2 `Conv1dBlock` stacked together with residual connection.
# @markdown `cond` is applied to `x` with [FiLM](https://arxiv.org/abs/1709.07871) conditioning.
import torch
import torch.nn as nn
from typing import Union
import math


'''
Network used for SRFMP
'''


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 cond_dim,
                 kernel_size=3,
                 n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self,
                 input_dim,
                 global_cond_dim,
                 diffusion_step_embed_dim=256,
                 down_dims=[32, 64, 128],
                 kernel_size=5,
                 n_groups=8
                 ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out * 2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )
        self.global_cond = None

    def forward(self,
                timestep: Union[torch.Tensor, float, int],
                sample: torch.Tensor):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1, -2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
            timesteps = timesteps.expand(sample.shape[0])
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps = timesteps.expand(sample.shape[0])
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 2:
            timesteps = timesteps.squeeze(1)

        global_feature = self.diffusion_step_encoder(timesteps)

        global_feature = torch.cat([global_feature, self.global_cond], axis=-1)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1, -2)
        # (B,T,C)
        return x


class StableUnet(nn.Module):
    def __init__(self,
                 input_dim,
                 global_cond_dim,
                 diffusion_step_embed_dim=256,
                 down_dims=[32, 64, 128],
                 kernel_size=5,
                 n_groups=8
                 ):
        super().__init__()
        self.unet = ConditionalUnet1D(input_dim=input_dim,
                                      global_cond_dim=global_cond_dim,
                                      diffusion_step_embed_dim=diffusion_step_embed_dim,
                                      down_dims=down_dims,
                                      kernel_size=kernel_size,
                                      n_groups=n_groups)
        self.input_dim = input_dim

    def forward(self, t, x):
        B = x.shape[0]
        dim = x.shape[1]
        x_ = x.reshape((B, dim // self.input_dim, self.input_dim))
        if len(t.shape) == 0:
            res = self.unet(t, x_)
        else:
            res = self.unet(t.squeeze(1), x_)
        return res.reshape((B, dim))


# this one learn the vector field of tau; the above one doesnt
class StableUnetLearnTau(nn.Module):
    def __init__(self,
                 input_dim,
                 global_cond_dim,
                 diffusion_step_embed_dim=256,
                 down_dims=[32, 64, 128],
                 kernel_size=5,
                 n_groups=8,
                 output_dim=32,
                 ):
        super().__init__()
        self.unet = ConditionalUnet1D(input_dim=input_dim,
                                      global_cond_dim=global_cond_dim,
                                      diffusion_step_embed_dim=diffusion_step_embed_dim,
                                      down_dims=down_dims,
                                      kernel_size=kernel_size,
                                      n_groups=n_groups)
        self.tau_vf = nn.Sequential(
            nn.Linear(1, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.input_dim = input_dim

    def forward(self, tau, z):
        x = z[..., :-1]
        tau = z[..., -1].unsqueeze(-1)
        B = x.shape[0]
        dim = x.shape[1]
        x_ = x.reshape((B, dim // self.input_dim, self.input_dim))
        dx = self.unet(tau, x_)
        dx = dx.reshape((B, dim))
        dtau = self.tau_vf(tau)
        return torch.hstack([dx, dtau])


class StableUnetLearnTauNew(nn.Module):
    def __init__(self,
                 input_dim,
                 global_cond_dim,
                 diffusion_step_embed_dim=256,
                 down_dims=[32, 64, 128],
                 kernel_size=5,
                 n_groups=8,
                 output_dim=32,
                 dtau_net_dim=512,
                 ):
        super().__init__()
        self.unet = ConditionalUnet1D(input_dim=input_dim,
                                      global_cond_dim=global_cond_dim,
                                      diffusion_step_embed_dim=diffusion_step_embed_dim,
                                      down_dims=down_dims,
                                      kernel_size=kernel_size,
                                      n_groups=n_groups)
        # previous 512
        self.tau_vf = nn.Sequential(
            nn.Linear(1 + global_cond_dim + output_dim, dtau_net_dim),
            nn.ReLU(),
            nn.Linear(dtau_net_dim, dtau_net_dim),
            nn.ReLU(),
            nn.Linear(dtau_net_dim, 1),
        )
        self.input_dim = input_dim

    def forward(self, tau, z):
        x = z[..., :-1]
        tau = z[..., -1].unsqueeze(-1)
        B = x.shape[0]
        dim = x.shape[1]
        x_ = x.reshape((B, dim // self.input_dim, self.input_dim))
        dx = self.unet(tau, x_)
        dx = dx.reshape((B, dim))
        global_cond = self.unet.global_cond
        tau_vf_input = torch.hstack([tau, x, global_cond])
        dtau = self.tau_vf(tau_vf_input)
        return torch.hstack([dx, dtau])


class StableUnetLearnTauStepEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 global_cond_dim,
                 diffusion_step_embed_dim=256,
                 down_dims=[32, 64, 128],
                 kernel_size=5,
                 n_groups=8,
                 output_dim=32,
                 dtau_net_dim=512,
                 ):
        super().__init__()
        self.unet = ConditionalUnet1D(input_dim=input_dim,
                                      global_cond_dim=global_cond_dim,
                                      diffusion_step_embed_dim=diffusion_step_embed_dim,
                                      down_dims=down_dims,
                                      kernel_size=kernel_size,
                                      n_groups=n_groups)
        dsed = diffusion_step_embed_dim
        self.tau_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        # previous 512
        self.tau_vf = nn.Sequential(
            nn.Linear(diffusion_step_embed_dim + global_cond_dim + output_dim, dtau_net_dim),
            nn.ReLU(),
            nn.Linear(dtau_net_dim, dtau_net_dim),
            nn.ReLU(),
            nn.Linear(dtau_net_dim, 1),
        )
        self.input_dim = input_dim

    def forward(self, tau, z):
        x = z[..., :-1]
        tau = z[..., -1].unsqueeze(-1)
        B = x.shape[0]
        dim = x.shape[1]
        x_ = x.reshape((B, dim // self.input_dim, self.input_dim))
        dx = self.unet(tau, x_)
        dx = dx.reshape((B, dim))
        global_cond = self.unet.global_cond
        tau_encode = self.tau_encoder(tau).squeeze(dim=1)
        tau_vf_input = torch.hstack([tau_encode, x, global_cond])
        dtau = self.tau_vf(tau_vf_input)
        return torch.hstack([dx, dtau])


if __name__ == '__main__':
    pred_horizon = 8
    action_dim = 2
    net = ConditionalUnet1D(action_dim, global_cond_dim=2)
    net.global_cond = torch.tensor([[0.5, 0.3] for _ in range(8)]).reshape(-1).unsqueeze(0)
    net.global_cond = torch.tensor([[0.5, 0.3], [0.5, 0.3]])
    noised_action = torch.randn((2, pred_horizon, action_dim))
    net.forward(torch.tensor([0.3, 0.5]), noised_action)

"""
Simple Unet.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# import ddpm
from .ddpm import ddpm_schedules


class TimeSiren(nn.Module):
    def __init__(self, emb_dim: int) -> None:
        super(TimeSiren, self).__init__()

        self.lin1 = nn.Linear(1, emb_dim, bias=False)
        self.lin2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1)
        # x = torch.sin(self.lin1(x))
        x = nn.ReLU()(self.lin1(x))
        x = self.lin2(x)
        return x


class MyConv1d(nn.Module):

    def __init__(
        self, in_channels, out_channels, filter_size, stride=1, bias=False, last=False
    ):
        """
        Args:
            in_channels: The number of input channels
            out_channels: The number of output channels
            filter_size: The size of the convolutional kernel
            stride: The stride (conv. ker. applied every stride pixels)
            bias: True for adding bias
            last: True if this is the last layer of the network
        """
        super().__init__()

        self.filter_size = filter_size
        self.stride = stride
        self.filter = nn.Parameter(torch.randn(out_channels, in_channels, filter_size))
        self.num_add = filter_size * in_channels   #it was self.filter.size(0) * self.filter.size(2), I changed it to this
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
            self.num_add += 1
        else:
            self.register_parameter("bias", None)
        self.last = last

    def forward(self, x):
        """
        Args:
            x: input, tensor of size (batch_size, in_channels, input_dim).

        Returns:
            The convolution of x with self.filter, tensor of size (batch_size, out_channels, out_dim),
            out_dim = (input_dim-filter_size)//stride+1
        """

        if self.last:
            return F.conv1d(x, self.filter, self.bias, stride=self.stride) / (
                self.num_add
            )
        else:
            return (
                F.conv1d(x, self.filter, self.bias, stride=self.stride)
                / self.num_add ** 0.5
            )


class MyConvTranspose1d(nn.Module):

    def __init__(self, in_channels, out_channels, filter_size, stride=1, bias=False):
        """
        Args:
            in_channels: The number of input channels
            out_channels: The number of output channels
            filter_size: The size of the convolutional kernel
            stride: The stride (conv. ker. applied every stride pixels)
            bias: True for adding bias
        """
        super().__init__()

        self.filter_size = filter_size
        self.stride = stride
        self.filter = nn.Parameter(torch.randn(in_channels, out_channels, filter_size))
        self.num_add = filter_size * in_channels
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
            self.num_add += 1
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        """
        Args:
            x: input, tensor of size (batch_size, in_channels, input_dim).

        Returns:
            The convolution of x with self.filter, tensor of size (batch_size, out_channels, out_dim),
            out_dim = (input_dim-filter_size)//stride+1
        """

        return (
            F.conv_transpose1d(x, self.filter, self.bias, stride=self.stride)
            / self.num_add ** 0.5
        )
    
class ChannelLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(ChannelLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape)

    def forward(self, x):
        # Permute the last two axes
        x = x.permute(0, 2, 1) # (batch_size, input_dim, in_channels)
        # Apply LayerNorm
        x = self.layer_norm(x)
        # Permute back to the original shape
        x = x.permute(0, 2, 1) # (batch_size, in_channels, input_dim)
        return x


class UnetDownTime(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, s: int, bias: bool) -> None:
        super(UnetDownTime, self).__init__()
        layers = [MyConv1d(in_channels, out_channels, s, s, bias), nn.ReLU()]
        self.model = nn.Sequential(*layers)
        self.timeembed = TimeSiren(in_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        timeembed = self.timeembed(t).view(-1, x.shape[1], 1)

        return self.model(x + timeembed)


class UnetUpTime(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, s: int, bias: bool) -> None:
        super(UnetUpTime, self).__init__()
        layers = [
            MyConvTranspose1d(2 * in_channels, out_channels, s, s, bias),
            nn.ReLU(),
        ]
        self.model = nn.Sequential(*layers)
        self.timeembed = TimeSiren(2 * in_channels)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat((x, skip), 1)
        timeembed = self.timeembed(t).view(-1, x.shape[1], 1)
        x = self.model(x + timeembed)

        return x


class hUNET(nn.Module):
    def __init__(
        self, input_dim, patch_size, in_channels, width, num_layers, bias=False
    ):

        super().__init__()

        receptive_field = patch_size**num_layers
        assert (
            input_dim % receptive_field == 0
        ), "patch_size**num_layers must divide input_dim!"

        self.downsize = nn.ModuleList(
            [
                UnetDownTime(in_channels, width, patch_size, bias=bias),
                *[
                    UnetDownTime(width, width, patch_size, bias=bias)
                    for _ in range(1, num_layers)
                ],
            ]
        )
        self.upsize = nn.ModuleList(
            [
                *[
                    UnetUpTime(width, width, patch_size, bias=bias)
                    for _ in range(1, num_layers)
                ],
                UnetUpTime(width, in_channels, patch_size, bias=bias),
            ]
        )
        self.readout = MyConv1d(2 * in_channels, in_channels, 1, 1, bias=bias)

    def forward(self, x, t):

        downsize_outs = [x]
        for module in self.downsize:
            x = module(x, t)
            downsize_outs.append(x)

        for module in self.upsize:
            x = module(x, downsize_outs.pop(), t)

        x = self.readout(torch.cat((x, downsize_outs.pop()), 1))

        return x


class UnetDownTimeChan(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, s: int, bias: bool) -> None:
        super(UnetDownTimeChan, self).__init__()
        layers = [MyConv1d(2 * in_channels, out_channels, s, s, bias), nn.ReLU()]
        self.model = nn.Sequential(*layers)
        self.timeembed = TimeSiren(in_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        timeembed = self.timeembed(t).unsqueeze(2).repeat(1, 1, x.shape[2])
        x = torch.cat((x, timeembed), 1)
        return self.model(x)


class UnetUpTimeChan(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, s: int, bias: bool) -> None:
        super(UnetUpTimeChan, self).__init__()
        layers = [
            MyConvTranspose1d(3 * in_channels, out_channels, s, s, bias),
            nn.ReLU(),
        ]
        self.model = nn.Sequential(*layers)
        self.timeembed = TimeSiren(in_channels)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        timeembed = self.timeembed(t).unsqueeze(2).repeat(1, 1, x.shape[2])
        x = torch.cat((x, skip, timeembed), 1)
        x = self.model(x)

        return x


class hUNETTimeChan(nn.Module):
    def __init__(
        self, input_dim, patch_size, in_channels, width, num_layers, bias=False
    ):

        super().__init__()

        receptive_field = patch_size**num_layers
        assert (
            input_dim % receptive_field == 0
        ), "patch_size**num_layers must divide input_dim!"

        self.downsize = nn.ModuleList(
            [
                UnetDownTimeChan(in_channels, width, patch_size, bias=bias),
                *[
                    UnetDownTimeChan(width, width, patch_size, bias=bias)
                    for _ in range(1, num_layers)
                ],
            ]
        )
        self.upsize = nn.ModuleList(
            [
                *[
                    UnetUpTimeChan(width, width, patch_size, bias=bias)
                    for _ in range(1, num_layers)
                ],
                UnetUpTimeChan(width, in_channels, patch_size, bias=bias),
            ]
        )
        self.readout = MyConv1d(2 * in_channels, in_channels, 1, 1, bias=bias)

    def forward(self, x, t):

        downsize_outs = [x]
        for module in self.downsize:
            x = module(x, t)
            downsize_outs.append(x)

        for module in self.upsize:
            x = module(x, downsize_outs.pop(), t)

        x = self.readout(torch.cat((x, downsize_outs.pop()), 1))

        return x


class UnetDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, s: int, bias: bool) -> None:
        super().__init__()
        layers = [MyConv1d(in_channels, out_channels, s, s, bias), nn.ReLU()]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, s: int, bias: bool) -> None:
        super().__init__()
        layers = [
            MyConvTranspose1d(2 * in_channels, out_channels, s, s, bias),
            nn.ReLU()
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class hUNETFullEmb(nn.Module):
    def __init__(
        self, input_dim, patch_size, in_channels, width, num_layers, bias=False
    ):

        super().__init__()

        receptive_field = patch_size**num_layers
        assert (
            input_dim % receptive_field == 0
        ), "patch_size**num_layers must divide input_dim!"

        self.embed_layer = nn.Sequential(
            nn.Linear(in_channels + 1, width),
            nn.ReLU(),
            nn.Linear(width, width),
        )

        self.downsize = nn.ModuleList(
            [
                UnetDown(width, width, patch_size, bias=bias),
                *[
                    UnetDown(width, width, patch_size, bias=bias)
                    for _ in range(1, num_layers)
                ],
            ]
        )
        self.upsize = nn.ModuleList(
            [
                *[
                    UnetUp(width, width, patch_size, bias=bias)
                    for _ in range(1, num_layers)
                ],
                UnetUp(width, width, patch_size, bias=bias),
            ]
        )
        self.readout = MyConv1d(2 * width, in_channels, 1, 1, bias=bias, last=True)

    def forward(self, x, t):

        t = torch.ones((x.shape[0], 1, x.shape[2]), device=x.device) * t.reshape(
            -1, 1, 1
        )
        x = torch.cat((x, t), 1).permute(0, 2, 1)
        x = self.embed_layer(x).permute(0, 2, 1)
        downsize_outs = [x]
        for module in self.downsize:
            x = module(x)
            downsize_outs.append(x)

        for module in self.upsize:
            x = module(x, downsize_outs.pop())

        x = self.readout(torch.cat((x, downsize_outs.pop()), 1))

        return x


class UnetDownBP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, s: int, bias: bool) -> None:
        super(UnetDownBP, self).__init__()
        layers = [
            MyConv1d(in_channels, out_channels, s, s, bias), 
            ChannelLayerNorm(out_channels),
            nn.GELU(),
        ]
        self.model = nn.Sequential(*layers)
        # self.timeembed = TimeSiren(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # timeembed = self.timeembed(t).view(-1, x.shape[1], 1)

        return self.model(x)


class UnetUpBP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, s: int, bias: bool) -> None:
        super(UnetUpBP, self).__init__()
        layers = [
            MyConvTranspose1d(2 * in_channels, out_channels, s, s, bias),
            ChannelLayerNorm(out_channels),
            nn.GELU(),
        ]
        self.model = nn.Sequential(*layers)
        # self.timeembed = TimeSiren(2*in_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x, skip), 1)
        # timeembed = self.timeembed(t).view(-1, x.shape[1], 1)
        x = self.model(x)

        return x


class bpTimeEmbed(nn.Module):
    def __init__(self, betas, n_T, process) -> None:
        super(bpTimeEmbed, self).__init__()
        self.n_T = n_T
        self.process = process
        noise_schedule = ddpm_schedules(betas[0], betas[1], n_T)
        # for k in ["sqrtab", "sqrtmab"]:
        for k in ["mean_over_var", "alphabar_t"]:
            self.register_buffer(k, noise_schedule[k])

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # xshape = (batch_size, v, input_dim)
        _ts = (t * self.n_T).long()
        # print(type(_ts))
        # print(self.mean_over_var[_ts])
        # proba = torch.exp(
        #     # self.sqrtab[_ts, None, None] * x / self.sqrtmab[_ts, None, None]**2
        #     x * self.mean_over_var[_ts, None, None]
        # )
        # proba = proba / proba.sum(1, keepdim=True)
        if self.process == "continuous":
            p_x0_given_xt = torch.softmax(x * self.mean_over_var[_ts, None, None], 1)
            # if any(t==1.0): print(proba, x, self.mean_over_var[_ts, None, None])
        elif self.process == "discrete":
            v = x.shape[1]
            mean = (self.alphabar_t * 1. + (1 - self.alphabar_t) / v).mean()
            std  = ((self.alphabar_t * 1. + (1 - self.alphabar_t) / v**2).mean() - mean**2)**0.5

            p_x0_given_xt = self.alphabar_t[_ts, None, None] * x + (
                1 - self.alphabar_t[_ts, None, None]
            ) / v * torch.ones_like(x)
            out = (p_x0_given_xt - mean) / std
        return out


class bpUNET(nn.Module):
    def __init__(
        self,
        input_dim,
        patch_size,
        in_channels,
        width,
        num_layers,
        bias=False,
        betas=(1e-4, 0.02),
        n_T=1000,
        **kwargs,
    ):

        super().__init__()

        receptive_field = patch_size**num_layers
        assert (
            input_dim % receptive_field == 0
        ), "patch_size**num_layers must divide input_dim!"

        self.timeembed = bpTimeEmbed(betas, n_T, kwargs['process'])

        self.embed_layer = nn.Sequential(MyConv1d(in_channels, width, filter_size=1, stride=1, bias=bias), nn.GELU())

        # list_widths = [width * (2**i) for i in range(1, num_layers+1)]
        list_widths = [width for _ in range(num_layers)]

        self.downsize = nn.ModuleList(
            [
                UnetDownBP(width, list_widths[0], patch_size, bias=bias),
                *[
                    UnetDownBP(list_widths[i-1], list_widths[i], patch_size, bias=bias)
                    for i in range(1, num_layers)
                ],
            ]
        )
        self.upsize = nn.ModuleList(
            [
                *[
                    UnetUpBP(list_widths[i], list_widths[i-1], patch_size, bias=bias)
                    for i in range(num_layers-1, 0, -1)
                ],
                UnetUpBP(list_widths[0], width, patch_size, bias=bias),
            ]
        )
        self.readout = MyConv1d(2 * width, in_channels, 1, 1, bias=bias, last=True)
        # self.readout = nn.Sequential(nn.Linear(2 * in_channels*input_dim, width*input_dim), nn.GELU(), nn.Linear(width*input_dim, in_channels*input_dim))

    def forward(self, x, t):
        # shape = x.shape
        x = self.timeembed(x, t) # BP time embedding
        x = self.embed_layer(x) # Embedding layer in width channels
        # print(x.shape)

        downsize_outs = [x]
        for module in self.downsize:
            x = module(x)
            downsize_outs.append(x)

        for module in self.upsize:
            x = module(x, downsize_outs.pop())

        x = self.readout(torch.cat((x, downsize_outs.pop()), 1))
        # x = x.view(shape)

        return x

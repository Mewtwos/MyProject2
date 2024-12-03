import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from torch import Tensor

from .norm_layers import LayerNorm, GRN
from .mfnet import *


class Block(nn.Module):
    """ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim: int, drop_path=0.0):
        super().__init__()
        self.dwconv: nn.Module = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.norm: nn.Module = LayerNorm(dim, eps=1e-6, data_format="channels_last")

        self.pwconv1: nn.Module = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act: nn.Module = nn.GELU()
        self.grn: nn.Module = GRN(4 * dim)
        self.pwconv2: nn.Module = nn.Linear(4 * dim, dim)
        self.drop_path: nn.Module = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXtV2_unet(nn.Module):
    """ConvNeXt V2

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
        self,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 6,
        depths: list[int] = None,
        dims: list[int] = [40, 80, 160, 320],
        drop_path_rate: float = 0.0,
        head_init_scale: float = 1.0,
        use_orig_stem: bool = False,
    ):
        super().__init__()
        self.use_orig_stem = False
        self.depths = depths
        if self.depths is None:  # set default value
            self.depths = [2, 2, 6, 2]
        self.patch_size = patch_size
        if dims is None:
            dims = [96, 192, 384, 768]
        self.use_orig_stem = use_orig_stem
        self.downsample_layers = (
            nn.ModuleList()
        )
        self.downsample_layers2 = (
            nn.ModuleList()
        )
        self.num_stage = len(depths)
        if self.use_orig_stem:
            self.stem_orig = nn.Sequential(
                nn.Conv2d(
                    in_chans,
                    dims[0],
                    kernel_size=patch_size // (2 ** (self.num_stage - 1)),
                    stride=patch_size // (2 ** (self.num_stage - 1)),
                ),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            )
        else:
            self.initial_conv = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=3, stride=1, padding=1),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
                nn.GELU(),
            )
            self.initial_conv2 = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=3, stride=1, padding=1),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
                nn.GELU(),
            )
            # depthwise conv for stem
            self.stem = nn.Sequential(
                nn.Conv2d(
                    dims[0],
                    dims[0],
                    kernel_size=patch_size // (2 ** (self.num_stage - 1)),
                    stride=patch_size // (2 ** (self.num_stage - 1)),
                    groups=dims[0],
                ),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            )
            self.stem2 = nn.Sequential(
                nn.Conv2d(
                    dims[0],
                    dims[0],
                    kernel_size=patch_size // (2 ** (self.num_stage - 1)),
                    stride=patch_size // (2 ** (self.num_stage - 1)),
                    groups=dims[0],
                ),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            )

        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
            self.downsample_layers2.append(downsample_layer)

        self.stages = (
            nn.ModuleList()
        ) 
        self.stages2 = (
            nn.ModuleList()
        ) 
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(self.num_stage):
            stage = nn.Sequential(
                *[
                    Block(dim=dims[i], drop_path=dp_rates[cur + j])
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            self.stages2.append(stage)
            cur += depths[i]

        self.apply(self._init_weights)

        #新增代码
        self.fpn1x = nn.Sequential(
            nn.ConvTranspose2d(dims[-1], dims[-1], kernel_size=2, stride=2),
            Norm2d(dims[-1]),
            # LayerNorm(dims[-1]),
            nn.GELU(),
            nn.ConvTranspose2d(dims[-1], dims[-1], kernel_size=2, stride=2),
        )
        self.fpn2x = nn.Sequential(
            nn.ConvTranspose2d(dims[-1], dims[-1], kernel_size=2, stride=2),
        )
        self.fpn3x = nn.Identity()
        self.fpn4x = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fpn1y = nn.Sequential(
            nn.ConvTranspose2d(dims[-1], dims[-1], kernel_size=2, stride=2),
            nn.BatchNorm2d(dims[-1]),
            # LayerNorm(dims[-1]),
            nn.GELU(),
            nn.ConvTranspose2d(dims[-1], dims[-1], kernel_size=2, stride=2),
        )
        self.fpn2y = nn.Sequential(
            nn.ConvTranspose2d(dims[-1], dims[-1], kernel_size=2, stride=2),
        )
        self.fpn3y = nn.Identity()
        self.fpn4y = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fusion1 = SEFusion(dims[-1], activation=nn.GELU())
        self.fusion2 = SEFusion(dims[-1], activation=nn.GELU())
        self.fusion3 = SEFusion(dims[-1], activation=nn.GELU())
        self.fusion4 = SEFusion(dims[-1], activation=nn.GELU())
        # self.encoder_sff = nn.ModuleList()
        # self.encoder_sff.append(SEFusion(dims[0]))
        # self.encoder_sff.append(SEFusion(dims[1]))
        # self.encoder_sff.append(SEFusion(dims[2]))
        # self.encoder_sff.append(SEFusion(dims[3])) 

        self.decoder = Decoder(nn.BatchNorm2d, (dims[-1], dims[-1], dims[-1], dims[-1]), 64, 0.1, 8, 6)
        # self.decoder = Decoder(LayerNorm, (dims[-1], dims[-1], dims[-1], dims[-1]), 64, 0.1, 8, 6)
        # self.decoder = Decoder((dims[-1], dims[-1], dims[-1], dims[-1]), 256, 0.1, 8, 6)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)
    
    def encoderx(self, x: Tensor):
        x = self.initial_conv(x)
        x = self.stem(x)
        x = self.stages[0](x)
        for i in range(3):
            x = self.downsample_layers[i](x)
            x = self.stages[i + 1](x)
        return x
    
    def encodery(self, x: Tensor):
        x = self.initial_conv2(x)
        x = self.stem2(x)
        x = self.stages2[0](x)
        for i in range(3):
            x = self.downsample_layers2[i](x)
            x = self.stages2[i + 1](x)
        return x
    
    def encoder(self, x: Tensor, y: Tensor):
        x = self.initial_conv(x)
        y = self.initial_conv2(y)
        x = self.stem(x)
        y = self.stem2(y)
        x = self.stages[0](x)
        y = self.stages2[0](y)
        x = self.encoder_sff[0](x, y)
        for i in range(3):
            x = self.downsample_layers[i](x)
            y = self.downsample_layers2[i](y)
            x = self.stages[i + 1](x)
            y = self.stages2[i + 1](y)
            x = self.encoder_sff[i + 1](x, y)
        return x, y
    
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = x.float()
        y = y.float()
        h, w = x.size(2), x.size(3)
        y = y.unsqueeze(1).repeat(1, 3, 1, 1)
        x = self.encoderx(x) # [b, 320, 16, 16]
        y = self.encodery(y)
        # x, y = self.encoder(x, y)
        res1x = self.fpn1x(x)
        res2x = self.fpn2x(x)
        res3x = self.fpn3x(x)
        res4x = self.fpn4x(x)
        res1y = self.fpn1x(y)
        res2y = self.fpn2x(y)
        res3y = self.fpn3x(y)
        res4y = self.fpn4x(y)
        res1 = self.fusion1(res1x, res1y)
        res2 = self.fusion2(res2x, res2y)
        res3 = self.fusion3(res3x, res3y)
        res4 = self.fusion4(res4x, res4y)
        x = self.decoder(res1, res2, res3, res4, h, w)
        return x
    
    def copy_parameters(self, source, target):
        for src_layer, tgt_layer in zip(source, target):
            # Ensure that both layers are the same type (e.g., Conv2d or LayerNorm)
            if isinstance(src_layer, nn.Conv2d) and isinstance(tgt_layer, nn.Conv2d):
                tgt_layer.weight.data.copy_(src_layer.weight.data)
                tgt_layer.bias.data.copy_(src_layer.bias.data)
            elif isinstance(src_layer, LayerNorm) and isinstance(tgt_layer, LayerNorm):
                tgt_layer.weight.data.copy_(src_layer.weight.data)
                tgt_layer.bias.data.copy_(src_layer.bias.data)
            elif isinstance(src_layer, nn.Linear) and isinstance(tgt_layer, nn.Linear):
                tgt_layer.weight.data.copy_(src_layer.weight.data)
                tgt_layer.bias.data.copy_(src_layer.bias.data)
            
    def copy_all_parameters(self):
        self.copy_parameters(self.initial_conv, self.initial_conv2)
        self.copy_parameters(self.stem, self.stem2)
        for i in range(3):
            self.copy_parameters(self.downsample_layers[i], self.downsample_layers2[i])
        for i in range(self.num_stage):
            self.copy_parameters(self.stages[i], self.stages2[i])


def convnextv2_unet_atto(**kwargs):
    model = ConvNeXtV2_unet(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model


def convnextv2_unet_femto(**kwargs):
    model = ConvNeXtV2_unet(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model


def convnextv2_unet_new(**kwargs):
    model = ConvNeXtV2_unet(depths=[2, 2, 6, 2], dims=[32, 64, 128, 256], **kwargs)
    return model


def convnextv2_unet_new2(**kwargs):
    model = ConvNeXtV2_unet(depths=[3, 3, 9, 3], dims=[40, 80, 160, 320], **kwargs)
    return model


def convnextv2_unet_pico(**kwargs):
    model = ConvNeXtV2_unet(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model


def convnextv2_unet_nano(**kwargs):
    model = ConvNeXtV2_unet(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model


def convnextv2_unet_tiny(**kwargs):
    model = ConvNeXtV2_unet(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model


def convnextv2_unet_base(**kwargs):
    model = ConvNeXtV2_unet(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model


def convnextv2_unet_large(**kwargs):
    model = ConvNeXtV2_unet(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model


def convnextv2_unet_huge(**kwargs):
    model = ConvNeXtV2_unet(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model

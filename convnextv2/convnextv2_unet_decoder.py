from typing import List, Tuple
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from torch import Tensor
from .norm_layers import LayerNorm, GRN
from .dwtconvfuse import DWTconvfuse
from .dwtaf import DWTAF
from .unetformer_decoder import Decoder

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
        ) 
        # self.norm: nn.Module = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.norm: nn.Module = nn.BatchNorm2d(dim, eps=1e-6)

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
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
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
        patch_size: int = 32,
        img_size: int = 128,
        in_chans: int = 3,
        num_classes: int = 1000,
        depths: list[int] = None,
        dims: list[int] = None,
        drop_path_rate: float = 0.0,
        head_init_scale: float = 1.0,
        use_orig_stem: bool = False,
    ):
        super().__init__()
        self.depths = depths
        if self.depths is None:  # set default value
            self.depths = [3, 3, 9, 3]
        self.img_size = img_size
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
                # LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
                nn.BatchNorm2d(dims[0], eps=1e-6),
                nn.GELU(),
            )
            self.initial_conv2 = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=3, stride=1, padding=1),
                # LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
                nn.BatchNorm2d(dims[0], eps=1e-6),
                nn.GELU(),
            )
            self.stem = nn.Sequential(
                nn.Conv2d(
                    dims[0],
                    dims[0],
                    kernel_size=patch_size // (2 ** (self.num_stage - 1)),
                    stride=patch_size // (2 ** (self.num_stage - 1)),
                    groups=dims[0],
                ),
                # LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
                nn.BatchNorm2d(dims[0], eps=1e-6),
            )
            self.stem2 = nn.Sequential(
                nn.Conv2d(
                    dims[0],
                    dims[0],
                    kernel_size=patch_size // (2 ** (self.num_stage - 1)),
                    stride=patch_size // (2 ** (self.num_stage - 1)),
                    groups=dims[0],
                ),
                # LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
                nn.BatchNorm2d(dims[0], eps=1e-6),
            )

        for i in range(3):
            downsample_layer = nn.Sequential(
                # LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.BatchNorm2d(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            downsample_layer2 = nn.Sequential(
                # LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.BatchNorm2d(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
            self.downsample_layers2.append(downsample_layer2)

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
            stage2 = nn.Sequential(
                *[
                    Block(dim=dims[i], drop_path=dp_rates[cur + j])
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            self.stages2.append(stage2)
            cur += depths[i]

        self.apply(self._init_weights)

        #新融合方案
        self.sff1 = DWTconvfuse(dims[0])
        self.sff2 = DWTconvfuse(dims[0])
        self.sff_stage = nn.ModuleList()
        self.sff_stage.append(DWTconvfuse(dims[1]))
        self.sff_stage.append(DWTconvfuse(dims[2]))

        self.dwtaf1 = DWTAF(num_layers=2, num_heads=16, hidden_size=dims[-1])
        # self.proj1 = nn.Sequential(
        #     nn.Conv2d(in_channels=dims[-1], out_channels=512, kernel_size=1,bias=False)
        # )
        self.decoder = Decoder(encoder_channels=dims, decode_channels=64, dropout=0.1, window_size=8, num_classes=6)
        

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, y:Tensor) -> Tensor:
        x = x.float()
        y = y.float()
        y = y.unsqueeze(1).repeat(1, 3, 1, 1)
        
        x = self.initial_conv(x)
        y = self.initial_conv2(y)
        x = self.sff1(x, y)
        res1 = x
        x = self.stem(x)
        y = self.stem2(y)
        x = self.sff2(x, y)
        res2 = x

        x = self.stages[0](x)
        y = self.stages2[0](y)

        res = []
        for i in range(3):
            x = self.downsample_layers[i](x)
            y = self.downsample_layers2[i](y)
            x = self.stages[i + 1](x)
            y = self.stages2[i + 1](y)
            if i < 2:
                x = self.sff_stage[i](x, y)
                res.append(x)
        
        h, w = x.shape[2], x.shape[3]
        x = x.view(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        y = y.view(y.shape[0], y.shape[1], -1).permute(0, 2, 1)
        x = self.dwtaf1(x, y)
        x = x.permute(0, 2, 1).view(x.shape[0], x.shape[2], h, w)

        #decoder
        res3 = res[0]
        res4 = res[1]
        res5 = x
        x = self.decoder(res2,res3,res4,res5,256,256)

        return x
            
    

def convnextv2_unet_atto(**kwargs):
    model = ConvNeXtV2_unet(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model


def convnextv2_unet_femto(**kwargs):
    model = ConvNeXtV2_unet(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
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

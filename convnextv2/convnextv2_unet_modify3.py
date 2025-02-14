import math
from typing import List, Tuple
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from torch import Tensor
from .norm_layers import LayerNorm, GRN
from .dwtconvfuse import DWTconvfuse
from .dwtaf import DWTAF
from .torch_wavelets import DWT_2D, IDWT_2D
from .decoder import Decoder

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x.size() 30,40,50,30
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 30,1,50,30
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)  # 30,1,50,30
        return self.sigmoid(x)  # 30,1,50,30
    
class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2*dim)
        
    def forward(self, x, y):
        q = self.q(x)
        k, v = self.kv(y).chunk(2, dim=-1)
        attn = (q @ k.transpose(-2,-1)) * (1.0/math.sqrt(q.size(-1)))
        attn = attn.softmax(dim=-1)
        return attn @ v

class DepthWiseConv(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(DepthWiseConv, self).__init__()
 
        # 逐通道卷积
        self.depth_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=in_channel,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_channel)
        #逐点卷积
        self.point_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    
    def forward(self,input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out
    
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
    
class FusionBlock(nn.Module): #共享stage
    def __init__(self, dim: int, drop_path=0.0, use_dwt=True):
        super().__init__()
        self.use_dwt = use_dwt
        self.blockx = Block(dim=dim, drop_path=drop_path)
        # self.blocky = Block(dim=dim, drop_path=drop_path)
        self.dwt = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')
        self.convx = nn.Conv2d(dim, dim // 4, kernel_size=1, bias=False)
        self.convy = nn.Conv2d(dim, dim // 4, kernel_size=1, bias=False)
        self.iconvx = nn.Conv2d(dim // 4, dim, kernel_size=1, bias=False)
        self.iconvy = nn.Conv2d(dim // 4, dim, kernel_size=1, bias=False)
        self.wx = torch.nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.wy = torch.nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.cross_attnx = CrossAttention(dim//4)
        self.cross_attny = CrossAttention(dim//4)
        self.spatial_attn = SpatialAttention()

    def forward(self, input) -> Tensor:
        x, y = input
        blockx = self.blockx(x)
        blocky = self.blockx(y)
        # blocky = self.blocky(y)
        if self.use_dwt:
            dwtx = self.dwt(self.convx(x))
            dwty = self.dwt(self.convy(y))
            llx, lhx, hlx, hhx = torch.split(dwtx, x.shape[1]//4, dim=1)
            lly, lhy, hly, hhy = torch.split(dwty, y.shape[1]//4, dim=1)
            # lh_fuse = torch.max(lhx, lhy)
            # hl_fuse = torch.max(hlx, hly)
            # hh_fuse = torch.max(hhx, hhy)

            #spatial attention
            spatial_attn = self.spatial_attn(torch.cat([lhx, hlx, hhx, lhy, hly, hhy], dim=1))
            llx = llx * spatial_attn
            llx = llx * spatial_attn

            # cross attention
            llxf = llx.flatten(2).transpose(1, 2)
            llyf = lly.flatten(2).transpose(1, 2)
            ll_fusex = self.cross_attnx(llxf, llyf)
            ll_fusey = self.cross_attny(llyf, llxf)
            ll_fusex = ll_fusex.transpose(1, 2).view_as(llx)
            ll_fusey = ll_fusey.transpose(1, 2).view_as(lly)

            dwtx = torch.cat([ll_fusex, lhx, hlx, hhx], dim=1)
            dwty = torch.cat([ll_fusey, lhy, hly, hhy], dim=1)

            # dwtx = torch.cat([llx, lh_fuse, hl_fuse, hh_fuse], dim=1)
            # dwty = torch.cat([lly, lh_fuse, hl_fuse, hh_fuse], dim=1)
            idwtx = self.idwt(dwtx)
            idwty = self.idwt(dwty)
            idwtx = self.iconvx(idwtx)
            idwty = self.iconvy(idwty)
            x = blockx + idwtx * self.wx + idwty * (1 - self.wx)
            y = blocky + idwty * self.wy + idwtx * (1 - self.wy)
            return x, y
        return blockx, blocky


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
        heatmap: bool = False,
        use_orig_stem: bool = False,
    ):
        super().__init__()
        self.depths = depths
        self.dsm_depths = [2, 2, 6, 2]
        self.heatmap = heatmap
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
            pass
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            nn.BatchNorm2d(dims[0], eps=1e-6)
        )
        stem2 = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            nn.BatchNorm2d(dims[0], eps=1e-6)   
        )
        self.downsample_layers.append(stem)
        self.downsample_layers2.append(stem2)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.BatchNorm2d(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            downsample_layer2 = nn.Sequential(
                nn.BatchNorm2d(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
            self.downsample_layers2.append(downsample_layer2)

        self.stages = (
            nn.ModuleList()
        )
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(self.num_stage):
            stage = nn.Sequential(
                *[
                    FusionBlock(dim=dims[i], drop_path=dp_rates[cur + j])
                    for j in range(depths[i])
                ],
                # FusionBlock(dim=dims[i], drop_path=dp_rates[cur + depths[i] - 1], use_dwt=True)
            )
            self.stages.append(stage)
            cur += depths[i]

        # self.head = nn.Conv2d(int(dims[0] / 2), num_classes, kernel_size=1, stride=1)

        # self.upsample_layers = nn.ModuleList()

        # for i in reversed(range(self.num_stage)):
        #     if i == 3:
        #         self.upsample_layers.append(
        #             UpsampleBlock(dims[i], int(dims[i] / 2), scale_factor=2)
        #         )
        #     elif i == 0:
        #         self.upsample_layers.append(
        #             UpsampleBlock(
        #                 dims[i] * 2,
        #                 int(dims[i]),
        #                 scale_factor=patch_size // (2 ** (self.num_stage - 1)),
        #             )
        #         )

        #         if self.use_orig_stem:
        #             pass
        #         else:
        #             self.initial_conv_upsample = UpsampleBlock(dims[i], int(dims[i] / 2), scale_factor=2)
        #     else:
        #         self.upsample_layers.append(
        #             UpsampleBlock(dims[i] * 2, int(dims[i] / 2), scale_factor=2)
        #         )

        self.apply(self._init_weights)

        #新融合方案
        self.decoder_dim = 256
        self.sff_stage = nn.ModuleList()
        self.sff_stage.append(DWTconvfuse(self.decoder_dim))
        self.sff_stage.append(DWTconvfuse(self.decoder_dim))
        self.sff_stage.append(DWTconvfuse(self.decoder_dim))
        self.sff_stage.append(DWTconvfuse(self.decoder_dim))

        self.fpn3x = nn.Sequential(
            nn.ConvTranspose2d(dims[-1], self.decoder_dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(self.decoder_dim),
            nn.GELU(),
            nn.ConvTranspose2d(self.decoder_dim, self.decoder_dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(self.decoder_dim),
            nn.GELU(),
            nn.ConvTranspose2d(self.decoder_dim, self.decoder_dim, kernel_size=2, stride=2),
        )
        self.fpn2x = nn.Sequential(
            nn.ConvTranspose2d(dims[-1], self.decoder_dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(self.decoder_dim),
            nn.GELU(),
            nn.ConvTranspose2d(self.decoder_dim, self.decoder_dim, kernel_size=2, stride=2),
        )
        self.fpn1x = nn.Sequential(
            nn.ConvTranspose2d(dims[-1], self.decoder_dim, kernel_size=2, stride=2),
        )
        self.fpn3y = nn.Sequential(
            nn.ConvTranspose2d(dims[-1], self.decoder_dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(self.decoder_dim),
            nn.GELU(),
            nn.ConvTranspose2d(self.decoder_dim, self.decoder_dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(self.decoder_dim),
            nn.GELU(),
            nn.ConvTranspose2d(self.decoder_dim, self.decoder_dim, kernel_size=2, stride=2),
        )
        self.fpn2y = nn.Sequential(
            nn.ConvTranspose2d(dims[-1], self.decoder_dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(self.decoder_dim),
            nn.GELU(),
            nn.ConvTranspose2d(self.decoder_dim, self.decoder_dim, kernel_size=2, stride=2),
        )
        self.fpn1y = nn.Sequential(
            nn.ConvTranspose2d(dims[-1], self.decoder_dim, kernel_size=2, stride=2),
        )
        self.convx = nn.Conv2d(dims[-1], self.decoder_dim, kernel_size=1,bias=False)
        self.convy = nn.Conv2d(dims[-1], self.decoder_dim, kernel_size=1,bias=False)
        # self.decoder = Decoder((256, 256, 256, 256), self.decoder_dim, dropout=0.1, window_size=8, num_classes=num_classes)
        self.decoder = Decoder((self.decoder_dim, self.decoder_dim, self.decoder_dim, self.decoder_dim), self.decoder_dim, dropout=0.1, window_size=8, num_classes=num_classes)
        
    def encoder(self, x: Tensor, y:Tensor) -> Tuple[Tensor, List[Tensor]]:
        h, w = x.shape[-2:]
        heatmaps = []
        for i in range(self.num_stage-1):
            x = self.downsample_layers[i](x)
            y = self.downsample_layers2[i](y)
            x, y = self.stages[i]((x, y))
            if self.heatmap:
                heatmaps.append(x)
        x = self.downsample_layers[-1](x)
        y = self.downsample_layers2[-1](y)
        if self.heatmap:
            heatmaps.append(x)

        res1x = self.convx(x)
        res1y = self.convy(y)
        res2x = self.fpn1x(x)
        res2y = self.fpn1y(y)
        res3x = self.fpn2x(x)
        res3y = self.fpn2y(y)
        res4x = self.fpn3x(x)
        res4y = self.fpn3y(y)
        res1 = self.sff_stage[-1](res1x, res1y)
        res2 = self.sff_stage[-2](res2x, res2y)
        res3 = self.sff_stage[-3](res3x, res3y)
        res4 = self.sff_stage[-4](res4x, res4y)

        out = self.decoder(res4, res3, res2, res1, h, w)
        if heatmaps:
            return out, heatmaps
        return out

    # def decoder(self, x: Tensor, enc_features: List[Tensor]):

    #     for i in range(3):
    #         x = self.upsample_layers[i](x)
    #         tmp = enc_features.pop()
    #         x = torch.cat([x, tmp], dim=1)
    #     x = self.upsample_layers[3](x)
    #     x = self.initial_conv_upsample(x)
    #     return x

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, y:Tensor) -> Tensor:
        x = x.float()
        y = y.float()
        y = y.unsqueeze(1).repeat(1, 3, 1, 1)
        out = self.encoder(x, y)
        # x = self.decoder(x, enc_features)
        # x = self.head(x)

        return out
            
    

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

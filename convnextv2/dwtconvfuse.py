import torch
from torch import nn
from .torch_wavelets import DWT_2D, IDWT_2D
from .wavelet import New_DWT_2D, New_IDWT_2D, Conv_DWT_2D, Conv_IDWT_2D

class DWTconvfuse(nn.Module):
    def __init__(self, channels, reduction=4):
        super(DWTconvfuse, self).__init__()
        self.convx = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        self.convy = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        # self.dwt = DWT_2D(wave='haar')
        # self.idwt = IDWT_2D(wave='haar')
        # self.dwt = New_DWT_2D(wave='sym2', in_channels=channels // reduction)#可更换小波
        # self.idwt = New_IDWT_2D(wave='sym2', in_channels=channels // reduction)
        # self.dwt = Conv_DWT_2D(wave='coif3', in_channels=channels // reduction)#统计参数量和计算复杂度
        # self.idwt = Conv_IDWT_2D(wave='coif3', out_channels=channels // reduction)
        self.dwt = nn.Conv2d(channels // reduction, channels, kernel_size=3, stride=2, padding=1)
        self.idwt = nn.ConvTranspose2d(channels, channels // reduction, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.finalconv = nn.Sequential(nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False))

    def forward(self, x, y):
        # final = x + y
        x1 = self.convx(x)
        y1 = self.convy(y)
        dwtx = self.dwt(x1)
        dwty = self.dwt(y1)
        llx, lhx, hlx, hhx = torch.split(dwtx, x1.shape[1], dim=1)
        lly, lhy, hly, hhy = torch.split(dwty, y1.shape[1], dim=1)
        lh_fuse = torch.max(lhx, lhy)
        hl_fuse = torch.max(hlx, hly)
        hh_fuse = torch.max(hhx, hhy)
        dwt = torch.cat([llx, lh_fuse, hl_fuse, hh_fuse], dim=1)
        final = self.idwt(dwt)
        final = self.finalconv(final) + x
        return final

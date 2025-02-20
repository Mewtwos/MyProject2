import torch
from torch import nn
from .torch_wavelets import DWT_2D, IDWT_2D

class DWTconvfuse(nn.Module):
    def __init__(self, channels, reduction=4):
        super(DWTconvfuse, self).__init__()
        self.convx = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        self.convy = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        self.dwt = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')
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

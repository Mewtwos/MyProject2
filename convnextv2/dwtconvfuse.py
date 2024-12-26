import torch
from torch import nn
from torch_wavelets import DWT_2D, IDWT_2D

class DWTconvfuse(nn.Module):
    def __init__(self, channels, kernel_size):
        super(DWTconvfuse, self).__init__()
        self.convx1 = nn.Conv2d(channels, channels, kernel_size, padding='same')
        self.convx2 = nn.Conv2d(4*channels, 4*channels, kernel_size, padding='same')
        self.convx3 = nn.Conv2d(16*channels, 16*channels, kernel_size, padding='same')
        self.convy1 = nn.Conv2d(channels, channels, kernel_size, padding='same')
        self.convy2 = nn.Conv2d(4*channels, 4*channels, kernel_size, padding='same')
        self.convy3 = nn.Conv2d(16*channels, 16*channels, kernel_size, padding='same')
        self.dwtx1 = DWT_2D(wave='haar')
        self.dwtx2 = DWT_2D(wave='haar')
        self.dwty1 = DWT_2D(wave='haar')
        self.dwty2 = DWT_2D(wave='haar')
        # self.idwt4 = IDWT_2D(wave='haar')
        self.idwt3 = IDWT_2D(wave='haar')
        self.idwt2 = IDWT_2D(wave='haar')
        self.idwt1 = IDWT_2D(wave='haar')
        self.finalconv = nn.Conv2d(channels*2, channels, 3, padding='same')

    def forward(self, x, y):
        x1 = self.convx1(x) # c * h * w
        wx1 = self.dwtx1(x) # 4c * h/2 * w/2
        x2 = self.convx2(wx1) # 4c * h/2 * w/2
        wx2 = self.dwtx2(wx1) # 16c * h/4 * w/4
        x3 = self.convx3(wx2) # 16c * h/4 * w/4
        y1 = self.convy1(y) # c * h * w
        wy1 = self.dwty1(y) # 4c * h/2 * w/2
        y2 = self.convy2(wy1) # 4c * h/2 * w/2
        wy2 = self.dwty2(wy1) # 16c * h/4 * w/4
        y3 = self.convy3(wy2) # 16c * h/4 * w/4
        fusexy3 = self.idwt3(x3 + y3) # 4c * h/2 * w/2
        x2 = self.idwt2(x2 + fusexy3) # c * h * w
        y2 = self.idwt1(y2 + fusexy3) # c * h * w
        x = x1 + x2
        y = y1 + y2
        return self.finalconv(torch.cat([x, y], dim=1))

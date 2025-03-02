from ptflops import get_model_complexity_info
import pywt
import pywt.data
import torch
import torch.nn.functional as F
import torch.nn as nn


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters

def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x

class New_DWT_2D(nn.Module):
    def __init__(self, wave, in_channels):
        super(New_DWT_2D, self).__init__()
        self.wt_filter, self.iwt_filter = create_wavelet_filter(wave, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)

    def forward(self, x):
        dwt = wavelet_transform(x, self.wt_filter)
        return dwt.view(x.shape[0], -1, x.shape[2]//2, x.shape[3]//2)

class New_IDWT_2D(nn.Module):
    def __init__(self, wave, in_channels):
        super(New_IDWT_2D, self).__init__()
        self.wt_filter, self.iwt_filter = create_wavelet_filter(wave, in_channels, in_channels, torch.float)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

    def forward(self, x):
        x = x.view(x.shape[0], -1, 4, x.shape[2], x.shape[3])
        idwt = inverse_wavelet_transform(x, self.iwt_filter)
        return idwt

class Conv_DWT_2D(nn.Module):
    def __init__(self, wave, in_channels):
        super(Conv_DWT_2D, self).__init__()
        kernel_size = len(pywt.Wavelet(wave).dec_lo)
        pad = (kernel_size // 2 - 1, kernel_size // 2 - 1)
        
        # 创建分解滤波器并设置卷积层
        dec_filters, _ = create_wavelet_filter(wave, in_channels, in_channels)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=4 * in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=pad,
            groups=in_channels,
            bias=False
        )
        self.conv.weight.data = dec_filters
        self.conv.weight.requires_grad = False

    def forward(self, x):
        return self.conv(x)
    
class Conv_IDWT_2D(nn.Module):
    def __init__(self, wave, out_channels):
        super(Conv_IDWT_2D, self).__init__()
        kernel_size = len(pywt.Wavelet(wave).rec_lo)
        pad = (kernel_size // 2 - 1, kernel_size // 2 - 1)
        
        # 创建重构滤波器并设置转置卷积层
        _, rec_filters = create_wavelet_filter(wave, out_channels, out_channels)
        self.conv_trans = nn.ConvTranspose2d(
            in_channels=4 * out_channels,  # 输入通道数 = 子带数 * 输出通道
            out_channels=out_channels,      # 恢复原始通道数
            kernel_size=kernel_size,
            stride=2,
            padding=pad,
            groups=out_channels,            # 每组处理4个子带
            bias=False
        )
        self.conv_trans.weight.data = rec_filters
        self.conv_trans.weight.requires_grad = False

    def forward(self, x):
        # # 输入形状: (b, c, 4, h, w)
        # b, c, _, h, w = x.shape
        # # 重塑为转置卷积需要的形状: (b, c*4, h, w)
        # x = x.view(b, c*4, h, w)
        return self.conv_trans(x)
    
if __name__ == "__main__":
    dwt = Conv_DWT_2D(wave='sym4', in_channels=768)
    input_res = (768, 256, 256)
    # 计算FLOPs和参数量
    macs, params = get_model_complexity_info(
        dwt, 
        input_res, 
        as_strings=True, 
        print_per_layer_stat=True,
        verbose=True
    )
    
    print(f"模型 FLOPs: {macs}")
    print(f"模型参数量: {params}")
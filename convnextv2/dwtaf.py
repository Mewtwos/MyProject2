import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import math
from torch.nn import LayerNorm, Linear, Dropout
from .torch_wavelets import DWT_2D, IDWT_2D


class WaveAttention(nn.Module):
    def __init__(self, sr_ratio=1, dim=512, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.dwt = DWT_2D(wave='haar')
        self.dwty = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')
        self.idwty = IDWT_2D(wave='haar')
        self.reduce = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(inplace=True),
        )
        self.reducey = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(inplace=True),
        )
        self.filter = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.filtery = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.kv_embed = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio) if sr_ratio > 1 else nn.Identity()
        self.kv_embedy = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio) if sr_ratio > 1 else nn.Identity()
        self.q = nn.Linear(dim, dim)
        self.qy = nn.Linear(dim, dim)
        self.kv = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2)
        )
        self.kvy = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2)
        )
        self.proj = nn.Linear(dim + dim // 4, dim)
        self.projy = nn.Linear(dim + dim // 4, dim)
        self.projcx = nn.Linear(dim + dim // 4, dim)
        self.projcy = nn.Linear(dim + dim // 4, dim)
        self.apply(self._init_weights)
        self.wx1 = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float32), requires_grad=True)
        self.wx2 = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float32), requires_grad=True)
        self.wy1 = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float32), requires_grad=True)
        self.wy2 = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float32), requires_grad=True)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, y):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        assert N == H * W
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        qy = self.qy(y).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        y = y.view(B, H, W, C).permute(0, 3, 1, 2)
        # x, pad = pad_to_fit(x, 8)
        # y, pady = pad_to_fit(y, 8)
        x_dwt = self.dwt(self.reduce(x))
        y_dwt = self.dwty(self.reducey(y))
        x_dwt = self.filter(x_dwt)
        y_dwt = self.filtery(y_dwt)
        x_idwt = self.idwt(x_dwt)
        y_idwt = self.idwty(y_dwt)
        # x_idwt = crop_to_original(x_idwt, pad).contiguous()
        # y_idwt = crop_to_original(y_idwt, pady).contiguous()
        x_idwt = x_idwt.view(B, -1, x_idwt.size(-2) * x_idwt.size(-1)).transpose(1, 2)
        y_idwt = y_idwt.view(B, -1, y_idwt.size(-2) * y_idwt.size(-1)).transpose(1, 2)

        kv = self.kv_embed(x_dwt).reshape(B, C, -1).permute(0, 2, 1)
        kvy = self.kv_embedy(y_dwt).reshape(B, C, -1).permute(0, 2, 1)
        kv = self.kv(kv).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        kvy = self.kvy(kvy).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        ky, vy = kvy[0], kvy[1]
        # self attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(torch.cat([x, x_idwt], dim=-1))

        attny = (qy @ ky.transpose(-2, -1)) * self.scale
        attny = attny.softmax(dim=-1)
        y = (attny @ vy).transpose(1, 2).reshape(B, N, C)
        y = self.projy(torch.cat([y, y_idwt], dim=-1))
        # cross attention
        c_x = (q @ ky.transpose(-2, -1)) * self.scale
        c_x = c_x.softmax(dim=-1)
        cx = (c_x @ vy).transpose(1, 2).reshape(B, N, C)
        cx = self.projcx(torch.cat([cx, y_idwt], dim=-1))

        c_y = (qy @ k.transpose(-2, -1)) * self.scale
        c_y = c_y.softmax(dim=-1)
        cy = (c_y @ v).transpose(1, 2).reshape(B, N, C)
        cy = self.projcy(torch.cat([cy, x_idwt], dim=-1))

        final_x = x * self.wx1 + cx * self.wx2
        final_y = y * self.wy1 + cy * self.wy2
        return final_x, final_y


class Mlp(nn.Module):
    def __init__(self, hidden_size=768, mlp_dim=3072, dropout_rate=0.1):
        super(Mlp, self).__init__()
        self.fc1 = Linear(hidden_size, mlp_dim)
        self.fc2 = Linear(mlp_dim, hidden_size)
        self.act_fn = nn.GELU()
        self.dropout = Dropout(dropout_rate)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, num_head=8, hidden_size=512):
        super(Block, self).__init__()
        self.hidden_size = hidden_size
        self.attention_norm = LayerNorm(hidden_size, eps=1e-6)
        self.attention_normd = LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn_normd = LayerNorm(hidden_size, eps=1e-6)
        self.ffn = Mlp(hidden_size=hidden_size)
        self.ffnd = Mlp(hidden_size=hidden_size)
        self.attn = WaveAttention(num_heads=num_head, dim=hidden_size)

    def forward(self, x, y):
        hx = x
        hy = y
        x = self.attention_norm(x)
        y = self.attention_normd(y)
        x, y = self.attn(x, y)
        x = x + hx
        y = y + hy

        hx = x
        hy = y
        x = self.ffn_norm(x)
        y = self.ffn_normd(y)
        x = self.ffn(x)
        y = self.ffnd(y)
        x = x + hx
        y = y + hy
        return x, y


class DWTAF(nn.Module):
    def __init__(self, num_layers=1, num_heads=16, hidden_size=512):
        super(DWTAF, self).__init__()
        self.num_layers = num_layers
        self.blocks = nn.ModuleList([Block(num_head=num_heads, hidden_size=hidden_size) for _ in range(num_layers)])
        self.normx = LayerNorm(hidden_size, eps=1e-6)
        self.normy = LayerNorm(hidden_size, eps=1e-6)

    def forward(self, x, y):
        for i in range(self.num_layers):
            x, y = self.blocks[i](x, y)
        x = self.normx(x)
        y = self.normy(y)
        return x + y




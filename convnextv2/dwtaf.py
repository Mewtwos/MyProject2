import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import math
from torch.nn import LayerNorm, Linear, Dropout
from .torch_wavelets import DWT_2D, IDWT_2D

def l2_norm(x):
    return torch.einsum("bcn, bn->bcn", x, 1 / torch.norm(x, p=2, dim=-2))

class DWTLinearAttention(nn.Module):
    def __init__(self, in_places, scale=8, eps=1e-6, mode="sa"):
        super(DWTLinearAttention, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.gamma_y = nn.Parameter(torch.zeros(1))
        self.gamma_cx = nn.Parameter(torch.zeros(1))
        self.gamma_cy = nn.Parameter(torch.zeros(1))
        self.in_places = in_places
        self.l2_norm = l2_norm
        self.eps = eps
        self.mode = mode

        self.query_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)
        self.query_conv_y = nn.Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv_y = nn.Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.value_conv_y = nn.Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

        # self.reduce = nn.Sequential(
        #     nn.Conv2d(in_places, in_places // 4, kernel_size=1, padding=0, stride=1),
        #     nn.BatchNorm2d(in_places // 4),
        #     nn.ReLU(inplace=True),
        # )
        # self.reducey = nn.Sequential(
        #     nn.Conv2d(in_places, in_places // 4, kernel_size=1, padding=0, stride=1),
        #     nn.BatchNorm2d(in_places // 4),
        #     nn.ReLU(inplace=True),
        # )
        self.dwt = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')

    def forward(self, x, y):
        batch_size, N, chnnels = x.shape
        width = height = int(math.sqrt(N))
        assert N == width * height
        x = x.view(batch_size, chnnels, width, height)
        y = y.view(batch_size, chnnels, width, height)
        x_dwt = self.dwt(x)
        y_dwt = self.dwt(y)
        llx, lhx, hlx, hhx = torch.split(x_dwt, x.shape[1], dim=1)
        lly, lhy, hly, hhy = torch.split(y_dwt, y.shape[1], dim=1)
        x = llx
        y = lly
        width = height = height // 2

        Q = self.query_conv(x).view(batch_size, -1, width * height)
        Qy = self.query_conv_y(y).view(batch_size, -1, width * height)
        # x = self.dwt(self.reduce(x))
        # x = self.filter(x)
        # idwt = self.idwt(x)
        K = self.key_conv(x).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)
        Ky = self.key_conv_y(y).view(batch_size, -1, width * height)
        Vy = self.value_conv_y(y).view(batch_size, -1, width * height)

        Q = self.l2_norm(Q).permute(-3, -1, -2)
        K = self.l2_norm(K)
        Qy = self.l2_norm(Qy).permute(-3, -1, -2)
        Ky = self.l2_norm(Ky)

        #self attention
        if self.mode == "sa":
            tailor_sum = 1 / (width * height + torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps))
            value_sum = torch.einsum("bcn->bc", V).unsqueeze(-1)
            value_sum = value_sum.expand(-1, chnnels, width * height)
            matrix = torch.einsum('bmn, bcn->bmc', K, V)
            matrix_sum = value_sum + torch.einsum("bnm, bmc->bcn", Q, matrix)
            weight_value = torch.einsum("bcn, bn->bcn", matrix_sum, tailor_sum)
            weight_value = weight_value.view(batch_size, chnnels, height, width)
            attention = (self.gamma * weight_value).contiguous()

            tailor_sumy = 1 / (width * height + torch.einsum("bnc, bc->bn", Qy, torch.sum(Ky, dim=-1) + self.eps))
            value_sumy = torch.einsum("bcn->bc", Vy).unsqueeze(-1)
            value_sumy = value_sumy.expand(-1, chnnels, width * height)
            matrixy = torch.einsum('bmn, bcn->bmc', Ky, Vy)
            matrix_sumy = value_sumy + torch.einsum("bnm, bmc->bcn", Qy, matrixy)
            weight_valuey = torch.einsum("bcn, bn->bcn", matrix_sumy, tailor_sumy)
            weight_valuey = weight_valuey.view(batch_size, chnnels, height, width)
            attentiony = (self.gamma_y * weight_valuey).contiguous()
            x_dwt = torch.cat([attention, lhx, hlx, hhx], dim=1)
            y_dwt = torch.cat([attentiony, lhy, hly, hhy], dim=1)
            attention = self.idwt(x_dwt)
            attentiony = self.idwt(y_dwt)
            attention = attention.view(batch_size, chnnels, -1).permute(0, 2, 1)
            attentiony = attentiony.view(batch_size, chnnels, -1).permute(0, 2, 1)
            return attention, attentiony
        #cross attention
        tailor_sum_cx = 1 / (width * height + torch.einsum("bnc, bc->bn", Q, torch.sum(Ky, dim=-1) + self.eps))
        value_sum = torch.einsum("bcn->bc", Vy).unsqueeze(-1)
        value_sum = value_sum.expand(-1, chnnels, width * height)
        matrixy = torch.einsum('bmn, bcn->bmc', Ky, Vy)
        matrix_sum_cx = value_sum + torch.einsum("bnm, bmc->bcn", Q, matrixy)
        weight_value_cx = torch.einsum("bcn, bn->bcn", matrix_sum_cx, tailor_sum_cx)
        weight_value_cx = weight_value_cx.view(batch_size, chnnels, height, width)
        attention_cx = (self.gamma_cx * weight_value_cx).contiguous()

        tailor_sum_cy = 1 / (width * height + torch.einsum("bnc, bc->bn", Qy, torch.sum(K, dim=-1) + self.eps))
        value_sum = torch.einsum("bcn->bc", V).unsqueeze(-1)
        value_sum = value_sum.expand(-1, chnnels, width * height)
        matrix = torch.einsum('bmn, bcn->bmc', K, V)
        matrix_sum_cy = value_sum + torch.einsum("bnm, bmc->bcn", Qy, matrix)
        weight_value_cy = torch.einsum("bcn, bn->bcn", matrix_sum_cy, tailor_sum_cy)
        weight_value_cy = weight_value_cy.view(batch_size, chnnels, height, width)
        attention_cy = (self.gamma_cy * weight_value_cy).contiguous()

        fuse_lh = torch.max(lhx, lhy)
        fuse_hl = torch.max(hlx, hly)
        fuse_hh = torch.max(hhx, hhy)
        final_x = torch.cat([attention_cx, fuse_lh, fuse_hl, fuse_hh], dim=1)
        final_y = torch.cat([attention_cy, fuse_lh, fuse_hl, fuse_hh], dim=1)
        final_x = self.idwt(final_x)
        final_y = self.idwt(final_y)
        final_x = final_x.view(batch_size, chnnels, -1).permute(0, 2, 1)
        final_y = final_y.view(batch_size, chnnels, -1).permute(0, 2, 1)
        return final_x, final_y
    
class LinearAttention(nn.Module):
    def __init__(self, in_places, scale=8, eps=1e-6, mode="sa"):
        super(LinearAttention, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.gamma_y = nn.Parameter(torch.zeros(1))
        self.gamma_cx = nn.Parameter(torch.zeros(1))
        self.gamma_cy = nn.Parameter(torch.zeros(1))
        self.in_places = in_places
        self.l2_norm = l2_norm
        self.eps = eps
        self.mode = mode

        self.query_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)
        self.query_conv_y = nn.Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv_y = nn.Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.value_conv_y = nn.Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

        if self.mode == "ca":
            self.wx1 = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float32), requires_grad=True)
            self.wx2 = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float32), requires_grad=True)
            self.wy1 = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float32), requires_grad=True)
            self.wy2 = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float32), requires_grad=True)
        # self.reduce = nn.Sequential(
        #     nn.Conv2d(in_places, in_places // 4, kernel_size=1, padding=0, stride=1),
        #     nn.BatchNorm2d(in_places // 4),
        #     nn.ReLU(inplace=True),
        # )
        # self.dwt = DWT_2D(wave='haar')
        # self.idwt = IDWT_2D(wave='haar')
        # self.filter = nn.Sequential(
        #     nn.Conv2d(in_places, in_places, kernel_size=3, padding=1, stride=1, groups=1),
        #     nn.BatchNorm2d(in_places),
        #     nn.ReLU(inplace=True),
        # )
        # self.proj = nn.Linear(in_places + in_places // 4, in_places)

    def forward(self, x, y):
        # Apply the feature map to the queries and keys
        # batch_size, chnnels, width, height = x.shape
        batch_size, N, chnnels = x.shape
        width = height = int(math.sqrt(N))
        assert N == width * height
        x = x.view(batch_size, chnnels, width, height)
        y = y.view(batch_size, chnnels, width, height)

        Q = self.query_conv(x).view(batch_size, -1, width * height)
        Qy = self.query_conv_y(y).view(batch_size, -1, width * height)
        # x = self.dwt(self.reduce(x))
        # x = self.filter(x)
        # idwt = self.idwt(x)
        K = self.key_conv(x).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)
        Ky = self.key_conv_y(y).view(batch_size, -1, width * height)
        Vy = self.value_conv_y(y).view(batch_size, -1, width * height)

        Q = self.l2_norm(Q).permute(-3, -1, -2)
        K = self.l2_norm(K)
        Qy = self.l2_norm(Qy).permute(-3, -1, -2)
        Ky = self.l2_norm(Ky)

        #self attention
        tailor_sum = 1 / (width * height + torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps))
        value_sum = torch.einsum("bcn->bc", V).unsqueeze(-1)
        value_sum = value_sum.expand(-1, chnnels, width * height)
        matrix = torch.einsum('bmn, bcn->bmc', K, V)
        matrix_sum = value_sum + torch.einsum("bnm, bmc->bcn", Q, matrix)
        weight_value = torch.einsum("bcn, bn->bcn", matrix_sum, tailor_sum)
        weight_value = weight_value.view(batch_size, chnnels, height, width)
        attention = (self.gamma * weight_value).contiguous()

        tailor_sumy = 1 / (width * height + torch.einsum("bnc, bc->bn", Qy, torch.sum(Ky, dim=-1) + self.eps))
        value_sumy = torch.einsum("bcn->bc", Vy).unsqueeze(-1)
        value_sumy = value_sumy.expand(-1, chnnels, width * height)
        matrixy = torch.einsum('bmn, bcn->bmc', Ky, Vy)
        matrix_sumy = value_sumy + torch.einsum("bnm, bmc->bcn", Qy, matrixy)
        weight_valuey = torch.einsum("bcn, bn->bcn", matrix_sumy, tailor_sumy)
        weight_valuey = weight_valuey.view(batch_size, chnnels, height, width)
        attentiony = (self.gamma_y * weight_valuey).contiguous()
        #cross attention
        if self.mode == "sa":
            attention = attention.view(batch_size, chnnels, width * height).permute(0, 2, 1)
            attentiony = attentiony.view(batch_size, chnnels, width * height).permute(0, 2, 1)
            return attention, attentiony
        tailor_sum_cx = 1 / (width * height + torch.einsum("bnc, bc->bn", Q, torch.sum(Ky, dim=-1) + self.eps))
        matrix_sum_cx = value_sum + torch.einsum("bnm, bmc->bcn", Q, matrixy)
        weight_value_cx = torch.einsum("bcn, bn->bcn", matrix_sum_cx, tailor_sum_cx)
        weight_value_cx = weight_value_cx.view(batch_size, chnnels, height, width)
        attention_cx = (self.gamma_cx * weight_value_cx).contiguous()

        tailor_sum_cy = 1 / (width * height + torch.einsum("bnc, bc->bn", Qy, torch.sum(K, dim=-1) + self.eps))
        matrix_sum_cy = value_sumy + torch.einsum("bnm, bmc->bcn", Qy, matrix)
        weight_value_cy = torch.einsum("bcn, bn->bcn", matrix_sum_cy, tailor_sum_cy)
        weight_value_cy = weight_value_cy.view(batch_size, chnnels, height, width)
        attention_cy = (self.gamma_cy * weight_value_cy).contiguous()

        final_x = attention * self.wx1 + attention_cx * self.wx2
        final_y = attentiony * self.wy1 + attention_cy * self.wy2
        final_x = final_x.view(batch_size, chnnels, width * height).permute(0, 2, 1)
        final_y = final_y.view(batch_size, chnnels, width * height).permute(0, 2, 1)
        # attention = attention.view(batch_size, width * height, chnnels)
        # idwt = idwt.view(batch_size, width * height, -1)
        # attention = self.proj(torch.cat([attention, idwt], dim=-1)).view(batch_size, chnnels, width, height)
        return final_x, final_y
    
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
    def __init__(self, num_head=8, hidden_size=512, mode="sa"):
        super(Block, self).__init__()
        self.hidden_size = hidden_size
        self.attention_norm = LayerNorm(hidden_size, eps=1e-6)
        self.attention_normd = LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn_normd = LayerNorm(hidden_size, eps=1e-6)
        self.ffn = Mlp(hidden_size=hidden_size)
        self.ffnd = Mlp(hidden_size=hidden_size)
        # self.attn = WaveAttention(num_heads=num_head, dim=hidden_size)
        self.attn = LinearAttention(in_places=hidden_size, mode=mode)

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
        # self.blocks = nn.ModuleList([Block(num_head=num_heads, hidden_size=hidden_size) for _ in range(num_layers)])
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            if i < 3 or i > 8:
                self.blocks.append(Block(num_head=num_heads, hidden_size=hidden_size, mode="sa"))
            else:
                self.blocks.append(Block(num_head=num_heads, hidden_size=hidden_size, mode="ca"))
        self.normx = LayerNorm(hidden_size, eps=1e-6)
        self.normy = LayerNorm(hidden_size, eps=1e-6)

    def forward(self, x, y):
        for i in range(self.num_layers):
            x, y = self.blocks[i](x, y)
        x = self.normx(x)
        y = self.normy(y)
        return x + y



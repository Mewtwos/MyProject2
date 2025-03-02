import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision.models as models
# from ptflops import get_model_complexity_info
import torch.utils.model_zoo as model_zoo
import math

__all__ = ['Res2Net', 'res2net50_v1b', 'res2net101_v1b']

model_urls = {
    'res2net50_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth',
    'res2net101_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_v1b_26w_4s-0812c246.pth',
}

class uc(nn.Module):
    def __init__(self, in_channels):
        super(uc, self).__init__()
        self.uc = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, in_channels // 2, 1, 1, 0)
        )

    def forward(self, x):
        return self.uc(x)


class HalfConv(nn.Module):
    def __init__(self, in_channels):
        super(HalfConv, self).__init__()
        self.halfconv = nn.Conv2d(in_channels, in_channels // 2, 1, 1, 0)

    def forward(self, x):
        return self.halfconv(x)


class Unified(nn.Module):
    def __init__(self, in_channels):
        super(Unified, self).__init__()
        self.uc = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(64, 64)),
            nn.Conv2d(in_channels, 64, 1, 1, 0)
        )

    def forward(self, x):
        return self.uc(x)


def upsample(x, size):
    return F.interpolate(x, size, mode='bilinear', align_corners=True)



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        mean_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([mean_out, max_out], dim=1)
        return x * self.sigmoid(self.conv1(out))


class GDBM(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GDBM, self).__init__()
        self.RGB_CBR1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.RGB_CBR2 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.DSM_Conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.DSM_Conv2_3x3 = nn.Conv2d(in_dim, in_dim, 3, 1, 1)
        self.DSM_Conv2_1x1 = nn.Conv2d(in_dim, out_dim, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()

        self.DConv1 = nn.Conv2d(out_dim, out_dim // 4, 3, 1, padding=1, dilation=1)
        self.DConv2 = nn.Conv2d(out_dim, out_dim // 4, 3, 1, padding=2, dilation=2)
        self.DConv4 = nn.Conv2d(out_dim, out_dim // 4, 3, 1, padding=4, dilation=4)
        self.DConv8 = nn.Conv2d(out_dim, out_dim // 4, 3, 1, padding=8, dilation=8)

        self.Conv = nn.Conv2d(in_dim, out_dim, 1, 1, 0)

    def forward(self, rgb, dsm):
        fr = self.RGB_CBR1(rgb) * self.RGB_CBR2(rgb)
        # print('fr:', fr.shape)  # [3, 512, 16, 16][3, 256, 32, 32][3, 128, 64, 64][3, 32, 64, 64]
        fd = self.DSM_Conv1(dsm) * self.sigmoid(self.DSM_Conv2_1x1(self.DSM_Conv2_3x3(dsm)))
        # print('fd:', fd.shape)  # [3, 512, 16, 16][3, 256, 32, 32][3, 128, 64, 64][3, 32, 64, 64]
        frd = self.Conv(torch.cat((fr, fd), dim=1))
        f = torch.cat((self.DConv1(frd), self.DConv2(frd), self.DConv4(frd), self.DConv8(frd)), dim=1)
        # print('f:', f.shape)  # [3, 512, 16, 16][3, 256, 32, 32][3, 128, 64, 64][3, 32, 64, 64]
        return f


class PRIM(nn.Module):
    def __init__(self):
        super(PRIM, self).__init__()
        self.S1 = SpatialAttention()
        self.S2 = SpatialAttention()
        self.S3 = SpatialAttention()
        self.S4 = SpatialAttention()

    def forward(self, m1, m2, m3, m4):
        p1 = self.S1(m1) + self.S2(m2)
        p2 = self.S2(m2) + self.S3(m3)
        p3 = self.S3(m3) + self.S4(m4)
        # AMM
        a1 = p1 * p2 + p1 + p2
        a2 = p2 * p3 + p2 + p3
        a = a1 * a2 + a1 + a2
        return a

class Res2Net(nn.Module):

    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = self.maxpool(x)

        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x5 = self.avgpool(x4)
        x6 = x5.view(x5.size(0), -1)
        x7 = self.fc(x6)

        return x7

class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
       
def res2net50_v1b(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b model.
    Res2Net-50 refers to the Res2Net-50_v1b_26w_4s.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_v1b_26w_4s'], map_location='cpu'))
    return model

class Res2Net_Ours(nn.Module):

    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        self.inplanes = 64
        super(Res2Net_Ours, self).__init__()

        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = self.maxpool(x)

        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x0, x1, x2, x3, x4
    
def res2net50_v1b_Ours(pretrained=True, **kwargs):
    """Constructs a Res2Net-50_v1b model.
    Res2Net-50 refers to the Res2Net-50_v1b_26w_4s.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net_Ours(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_v1b_26w_4s']))
    return model

def res2net101_v1b(pretrained=True, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net101_v1b_26w_4s']))
    return model

def res2net101_v1b_Ours(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net_Ours(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net101_v1b_26w_4s']))
    return model

def Res2Net_model(ind=101):
    if ind == 50:
        model_base = res2net50_v1b(pretrained=True)
        model = res2net50_v1b_Ours()

    if ind == 101:
        model_base = res2net101_v1b(pretrained=False)
        model = res2net101_v1b_Ours()

    pretrained_dict = model_base.state_dict()
    model_dict = model.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model

class FFNet(nn.Module):
    def __init__(self, num_classes=6, ind=50, pretrained=True):
        super(FFNet, self).__init__()
        # Backbone model
        self.layer_rgb = Res2Net_model(ind)
        self.layer_dsm = Res2Net_model(ind)
        self.trans = nn.Conv2d(1, 3, 1, 1)

        # Fusion Module
        self.fu_1 = GDBM(64, 32)
        self.fu_2 = GDBM(256, 128)
        self.fu_3 = GDBM(512, 256)
        self.fu_4 = GDBM(1024, 512)

        self.rgb2_conv = nn.Conv2d(256, 64, 1, 1, 0)
        self.dsm2_conv = nn.Conv2d(256, 64, 1, 1, 0)

        self.UC_R_2048 = uc(2048)
        self.UC_R_1024 = uc(1024)
        self.UC_R_512 = uc(512)
        self.UC_D_2048 = uc(2048)
        self.UC_D_1024 = uc(1024)
        self.UC_D_512 = uc(512)

        self.half_R_64 = HalfConv(64)
        self.half_R_128 = HalfConv(128)
        self.half_R_256 = HalfConv(256)
        self.half_R_512 = HalfConv(512)
        self.half_R_1024 = HalfConv(1024)
        self.half_D_64 = HalfConv(64)
        self.half_D_128 = HalfConv(128)
        self.half_D_256 = HalfConv(256)
        self.half_D_512 = HalfConv(512)
        self.half_D_1024 = HalfConv(1024)

        self.uni_R_32 = Unified(32)
        self.uni_R_128 = Unified(128)
        self.uni_R_256 = Unified(256)
        self.uni_R_512 = Unified(512)
        self.uni_D_32 = Unified(32)
        self.uni_D_128 = Unified(128)
        self.uni_D_256 = Unified(256)
        self.uni_D_512 = Unified(512)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        # Decoder
        self.decoder1 = PRIM()
        self.decoder2 = PRIM()
        self.d1_CBR = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.d2_CBR = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.predict = nn.Conv2d(16, num_classes, 1, 1, 0)

    def forward(self, rgb, dsm):
        # x = torch.chunk(rgb, 4, dim=1)
        # rgb = torch.cat(x[0:3], dim=1)
        # dsm = x[3]
        dsm = dsm.unsqueeze(1)

        # Encoder
        rgb_1, rgb_2, rgb_3, rgb_4, rgb_5 = self.layer_rgb(rgb)
        dsm_1, dsm_2, dsm_3, dsm_4, dsm_5 = self.layer_dsm(self.trans(dsm))

        # print('rgb_1:', rgb_1.shape)
        # print('dsm_5:', dsm_5.shape)
        # rgb1,dsm1: [3, 64, 64, 64]
        # rgb2,dsm2: [3, 256, 64, 64]
        # rgb3,dsm3: [3, 512, 32, 32]
        # rgb4,dsm4: [3, 1024, 16, 16]
        # rgb5,dsm5: [3, 2048, 8, 8]

        rgb_12 = rgb_1 + self.rgb2_conv(rgb_2)
        rgb_23 = rgb_2 + self.UC_R_512(rgb_3)
        rgb_34 = rgb_3 + self.UC_R_1024(rgb_4)
        rgb_45 = rgb_4 + self.UC_R_2048(rgb_5)
        dsm_12 = dsm_1 + self.dsm2_conv(dsm_2)
        dsm_23 = dsm_2 + self.UC_D_512(dsm_3)
        dsm_34 = dsm_3 + self.UC_D_1024(dsm_4)
        dsm_45 = dsm_4 + self.UC_D_2048(dsm_5)

        # print('rgb_45:', rgb_45.shape)
        # print('dsm_45:', dsm_45.shape)
        # rgb_12,dsm_12: [3, 64, 64, 64]
        # rgb_23,dsm_23: [3, 256, 64, 64]
        # rgb_34,dsm_34: [3, 512, 32, 32]
        # rgb_45,dsm_45: [3, 1024, 16, 16]

        fu_4 = self.fu_4(rgb_45, dsm_45)  # [3, 512, 16, 16]
        # print('fu_4:', fu_4.shape)
        a3 = rgb_34 + self.up2(fu_4)  # [3, 512, 32, 32]
        b3 = dsm_34 + self.up2(fu_4)  # [3, 512, 32, 32]
        fu_3 = self.fu_3(a3, b3)  # [3, 256, 32, 32]
        # print('fu_3:', fu_3.shape)
        a2 = rgb_23 + self.up2(fu_3)  # [3, 256, 64, 64]
        b2 = dsm_23 + self.up2(fu_3)  # [3, 256, 64, 64]
        fu_2 = self.fu_2(a2, b2)  # [3, 128, 64, 64]
        # print('fu_2:', fu_2.shape)
        a1 = rgb_12 + self.half_R_128(fu_2)  # [3, 64, 64, 64]
        b1 = dsm_12 + self.half_D_128(fu_2)  # [3, 64, 64, 64]
        fu_1 = self.fu_1(a1, b1)  # [3, 32, 64, 64]
        # print('fu_1:', fu_1.shape)

        rf1 = self.half_R_64(rgb_12) * fu_1  # [3, 32, 64, 64]
        rf1 = self.uni_R_32(rf1)  # [3, 64, 64, 64]
        # print('rf1:', rf1.shape)
        rf2 = self.half_R_256(rgb_23) * fu_2  # [3, 128, 64, 64]
        rf2 = self.uni_R_128(rf2)  # [3, 64, 64, 64]
        # print('rf2:', rf2.shape)
        rf3 = self.half_R_512(rgb_34) * fu_3  # [3, 256, 32, 32]
        rf3 = self.uni_R_256(rf3)  # [3, 64, 64, 64]
        # print('rf3:', rf3.shape)
        rf4 = self.half_R_1024(rgb_45) * fu_4  # [3, 512, 16, 16]
        rf4 = self.uni_R_512(rf4)  # [3, 64, 64, 64]
        # print('rf4:', rf4.shape)

        df1 = self.half_D_64(dsm_12) * fu_1  # [3, 32, 64, 64]
        df1 = self.uni_D_32(df1)  # [3, 64, 64, 64]
        # print('df1:', df1.shape)
        df2 = self.half_D_256(dsm_23) * fu_2  # [3, 128, 64, 64]
        df2 = self.uni_D_128(df2)  # [3, 64, 64, 64]
        # print('df2:', df2.shape)
        df3 = self.half_D_512(dsm_34) * fu_3  # [3, 256, 32, 32]
        df3 = self.uni_D_256(df3)  # [3, 64, 64, 64]
        # print('df3:', df3.shape)
        df4 = self.half_D_1024(dsm_45) * fu_4  # [3, 512, 16, 16]
        df4 = self.uni_D_512(df4)  # [3, 64, 64, 64]
        # print('df4:', df4.shape)

        # Decoder
        d1 = self.decoder1(rf1, rf2, rf3, rf4)  # [3, 64, 64, 64]
        # print('d1:', d1.shape)
        d2 = self.decoder2(df1, df2, df3, df4)  # [3, 64, 64, 64]
        # print('d2:', d2.shape)

        final_output = self.predict(self.up4(self.d1_CBR(d1)) + self.up4(self.d2_CBR(d2)))

        return final_output


if __name__ == "__main__":
    image = torch.randn(3, 3, 256, 256)
    ndsm = torch.randn(3, 1, 256, 256)
    model = FFNet(num_classes=6)
    # print(model)
    out = model(image, ndsm)
    print(out.shape)
#     image = torch.randn( 3, 256, 64, 64)
#     traspose = T(256)
#     out = traspose(image)
#     print(out.shape)
#     flops, params = get_model_complexity_info(model,
#       (4, 256, 256), as_strings=True, print_per_layer_stat=True, verbose=True)
#     print(params)  # [3, 6, 256, 256]
#     print(flops)



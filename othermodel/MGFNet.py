import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch._utils
from torch.hub import load_state_dict_from_url
BN_MOMENTUM = 0.1
import torch.utils.model_zoo as model_zoo


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)

        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)

        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self,in_channels, block, layers, num_classes=1000):

        self.inplanes = 64
        super(ResNet, self).__init__()
        # 600,600,3 -> 300,300,64
        self.conv1  = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1    = nn.BatchNorm2d(64)
        self.relu   = nn.ReLU(inplace=True)
        # 300,300,64 -> 150,150,64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
        # 150,150,64 -> 150,150,256
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 150,150,256 -> 75,75,512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 75,75,512 -> 38,38,1024
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # 38,38,1024 -> 19,19,2048
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        x       = self.conv1(x)
        x       = self.bn1(x)
        feat1   = self.relu(x)

        x       = self.maxpool(feat1)
        feat2   = self.layer1(x)

        feat3   = self.layer2(feat2)
        feat4   = self.layer3(feat3)
        feat5   = self.layer4(feat4)
        return [feat1, feat2, feat3, feat4, feat5]

def resnet50(pretrained=False,in_channels=3, **kwargs):
    model = ResNet(in_channels=in_channels, block=Bottleneck, layers=[3, 4, 6, 3], **kwargs)
    if pretrained:

        pretrained_dict = model_zoo.load_url('https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth', model_dir='pretrains')

        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['conv1.weight']}
        print('loadding pretrained model')

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    del model.avgpool
    del model.fc
    return model

def resnet101(pretrained=False,in_channels=3, **kwargs):

    model = ResNet(in_channels=in_channels, block=Bottleneck, layers=[3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(
            model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth', model_dir='pretrains'),
            strict=False)
        
    print('loadding pretrained model:ResNet101')

    del model.avgpool
    del model.fc
    return model

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1  = conv3x3(inplanes, planes, stride)
        self.bn1    = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu   = nn.ReLU(inplace=True)
        self.conv2  = conv3x3(planes, planes)
        self.bn2    = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1  = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1    = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2  = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2    = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3  = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3    = nn.BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu   = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out

class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self.num_inchannels = num_inchannels
        self.num_branches   = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches       = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers    = self._make_fuse_layers()
        self.relu           = nn.ReLU(inplace=True)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index], num_channels[branch_index] * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for branch_index in range(num_branches):
            branches.append(self._make_one_branch(branch_index, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        fuse_layers = []
        for i in range(self.num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(self.num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(self.num_inchannels[j], self.num_inchannels[i], 1, 1, 0, bias=False),
                        nn.BatchNorm2d(self.num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(self.num_inchannels[j], self.num_inchannels[i], 3, 2, 1, bias=False),
                                    nn.BatchNorm2d(self.num_inchannels[i], momentum=BN_MOMENTUM)
                                )
                            )
                        else:
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(self.num_inchannels[j], self.num_inchannels[j], 3, 2, 1, bias=False),
                                    nn.BatchNorm2d(self.num_inchannels[j], momentum=BN_MOMENTUM),
                                    nn.ReLU(inplace=True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = 0
            # y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(0, self.num_branches):
                if j > i:
                    width_output    = x[i].shape[-1]
                    height_output   = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=True
                    )
                elif i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


class HighResolutionNet_Classification(nn.Module):
    def __init__(self, num_classes, backbone):
        super(HighResolutionNet_Classification, self).__init__()
        num_filters = {
            'hrnetv2_w18' : [18, 36, 72, 144],
            'hrnetv2_w32' : [32, 64, 128, 256],
            'hrnetv2_w48' : [48, 96, 192, 384],
        }[backbone]
        # stem net
        self.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2  = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2    = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu   = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)

        pre_stage_channels              = [Bottleneck.expansion * 64]
        num_channels                    = [num_filters[0], num_filters[1]]
        self.transition1                = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage2, pre_stage_channels = self._make_stage(1, 2, BasicBlock, [4, 4], num_channels, num_channels)

        num_channels                    = [num_filters[0], num_filters[1], num_filters[2]]
        self.transition2                = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(4, 3, BasicBlock, [4, 4, 4], num_channels, num_channels)

        num_channels                    = [num_filters[0], num_filters[1], num_filters[2], num_filters[3]]
        self.transition3                = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(3, 4, BasicBlock, [4, 4, 4, 4], num_channels, num_channels)

        self.pre_stage_channels         = pre_stage_channels

        self.incre_modules, self.downsamp_modules, self.final_layer = self._make_head(Bottleneck, pre_stage_channels)

        self.classifier = nn.Linear(2048, num_classes)

    def _make_layer(self, block, inplanes, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_transition_layer(self, num_inchannels, num_channels):
        num_branches_pre = len(num_inchannels)
        num_branches_cur = len(num_channels)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels[i] != num_inchannels[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[i], num_channels[i], 3, 1, 1, bias=False),
                        nn.BatchNorm2d(num_channels[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)
                    ))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = [
                    nn.Sequential(
                        nn.Conv2d(num_inchannels[-1], num_channels[i], 3, 2, 1, bias=False),
                        nn.BatchNorm2d(num_channels[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)
                    )
                ]
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_stage(self, num_modules, num_branches, block, num_blocks, num_inchannels, num_channels, multi_scale_output=True):
        modules = []
        for i in range(num_modules):
            modules.append(
                HighResolutionModule(num_branches, block, num_blocks, num_inchannels, num_channels, multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def _make_head(self, block, pre_stage_channels):
        head_channels = [32, 64, 128, 256]

        incre_modules = []
        for i, channels  in enumerate(pre_stage_channels):
            incre_module = self._make_layer(block, channels, head_channels[i], 1, stride=1)
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)
            
        downsamp_modules = []
        for i in range(len(pre_stage_channels)-1):
            in_channels = head_channels[i] * block.expansion
            out_channels = head_channels[i+1] * block.expansion

            downsamp_module = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            )

            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=head_channels[3] * block.expansion,
                out_channels=2048,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(2048, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )

        return incre_modules, downsamp_modules, final_layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
        
        x_list = []
        for i in range(2):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(3):
            if self.transition2[i] is not None:
                if i < 2:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(4):
            if self.transition3[i] is not None:
                if i < 3:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        y = self.incre_modules[0](y_list[0])
        for i in range(len(self.downsamp_modules)):
            y = self.incre_modules[i+1](y_list[i+1]) + \
                        self.downsamp_modules[i](y)

        y = self.final_layer(y)

        if torch._C._get_tracing_state():
            y = y.flatten(start_dim=2).mean(dim=2)
        else:
            y = F.avg_pool2d(y, kernel_size=y.size()
                                 [2:]).view(y.size(0), -1)

        y = self.classifier(y)

        return y
        
def hrnet_classification(pretrained=False, backbone='hrnetv2_w18'):
    model = HighResolutionNet_Classification(num_classes=1000, backbone=backbone)
    if pretrained:
        model_urls = {
            'hrnetv2_w18' : "https://github.com/bubbliiiing/hrnet-pytorch/releases/download/v1.0/hrnetv2_w18_imagenet_pretrained.pth",
            'hrnetv2_w32' : "https://github.com/bubbliiiing/hrnet-pytorch/releases/download/v1.0/hrnetv2_w32_imagenet_pretrained.pth",
            'hrnetv2_w48' : "https://github.com/bubbliiiing/hrnet-pytorch/releases/download/v1.0/hrnetv2_w48_imagenet_pretrained.pth",
        }
        state_dict = load_state_dict_from_url(model_urls[backbone], model_dir="pretrains")
        model.load_state_dict(state_dict)
    print('loadding pretrained model:HRNet')
    return model

class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        return y

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MGFM(nn.Module):
    def __init__(self, in_channels):
        super(MGFM, self).__init__()
        self.in_channel = in_channels
        self.eca_x = eca_block(channel=self.in_channel)
        self.eca_y = eca_block(channel=self.in_channel)
        self.mlp_x = Mlp(in_features=self.in_channel * 2, out_features=self.in_channel)
        self.mlp_y = Mlp(in_features=self.in_channel * 2, out_features=self.in_channel)
        self.sigmoid = nn.Sigmoid()

        self.mlp = Mlp(in_features= in_channels,out_features=in_channels)

    def forward(self, opt, sar):

        # Fusion-Stage-1 ECA Channel Attention
        w_opt = self.eca_x(opt)
        w_sar = self.eca_y(sar)
        N, C, H, W = w_opt.shape

        w_opt = torch.flatten(w_opt, 1)
        w_sar = torch.flatten(w_sar, 1)

        w = torch.concat([w_opt, w_sar], 1)
        
        # Fusion-Stage-2 MLP
        w1 = self.mlp_x(w)
        w1 = self.sigmoid(w1.reshape([N, self.in_channel, H, W]))

        w2 = self.mlp_y(w)
        w2 = self.sigmoid(w2.reshape([N, self.in_channel, H, W]))

        # Gating-Stage
        out1 = opt * w1
        out2 = sar * w2
        f = torch.cat((out1,out2),1)

        return f

class HRnet_Backbone(nn.Module):
    def __init__(self, backbone = 'hrnetv2_w18', pretrained = False):
        super(HRnet_Backbone, self).__init__()
        self.model    = hrnet_classification(backbone = backbone, pretrained = pretrained)
        del self.model.incre_modules
        del self.model.downsamp_modules
        del self.model.final_layer
        del self.model.classifier
        

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.conv2(x)
        x = self.model.bn2(x)
        x = self.model.relu(x)
        x = self.model.layer1(x)
        
        x_list = []
        for i in range(2):
            if self.model.transition1[i] is not None:
                x_list.append(self.model.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.model.stage2(x_list)

        x_list = []
        for i in range(3):
            if self.model.transition2[i] is not None:
                if i < 2:
                    x_list.append(self.model.transition2[i](y_list[i]))
                else:
                    x_list.append(self.model.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.model.stage3(x_list)

        x_list = []
        for i in range(4):
            if self.model.transition3[i] is not None:
                if i < 3:
                    x_list.append(self.model.transition3[i](y_list[i]))
                else:
                    x_list.append(self.model.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.model.stage4(x_list)
        
        return y_list

class MGFNet(nn.Module):
    def __init__(self, num_classes = 21, backbone = 'hrnetv2_w48', pretrained = False):
        super(MGFNet, self).__init__()
        self.opt_encoder       = HRnet_Backbone(backbone = backbone, pretrained = pretrained)
        self.sar_encoder       = resnet101(pretrained)
        last_inp_channels   = np.sum(self.opt_encoder.model.pre_stage_channels)

        self.last_layer = nn.Sequential(
            nn.Conv2d(in_channels=last_inp_channels, out_channels=last_inp_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=last_inp_channels, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        )

        layer = [24, 48, 96, 192]
        self.fusion_1 = MGFM(layer[0])
        self.fusion_2 = MGFM(layer[1])
        self.fusion_3 = MGFM(layer[2])
        self.fusion_4 = MGFM(layer[3])

        self.shortcut_conv1 = nn.Sequential(
            nn.Conv2d(384, 192, 1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        self.shortcut_conv2 = nn.Sequential(
            nn.Conv2d(192, 96, 1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        self.shortcut_conv3 = nn.Sequential(
            nn.Conv2d(96, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.shortcut_conv4 = nn.Sequential(
            nn.Conv2d(48, 24, 1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        self.shortcut_conv1_1 = nn.Sequential(
            nn.Conv2d(2048, 192, 1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        self.shortcut_conv2_1 = nn.Sequential(
            nn.Conv2d(1024, 96, 1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        self.shortcut_conv3_1 = nn.Sequential(
            nn.Conv2d(512, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.shortcut_conv4_1 = nn.Sequential(
            nn.Conv2d(256, 24, 1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
    def forward(self, opt,sar):
        sar = sar.unsqueeze(1).repeat(1, 3, 1, 1)
        H, W = opt.size(2), opt.size(3)

        # MFE Network
        [feat1_opt, feat2_opt, feat3_opt, feat4_opt] = self.opt_encoder(opt)
        [feat1_sar, feat2_sar, feat3_sar, feat4_sar, feat5_sar] = self.sar_encoder(sar)

        # Shortcut Layer
        feat4_opt = self.shortcut_conv1(feat4_opt)
        feat3_opt = self.shortcut_conv2(feat3_opt)
        feat2_opt = self.shortcut_conv3(feat2_opt)
        feat1_opt = self.shortcut_conv4(feat1_opt)

        feat5_sar = self.shortcut_conv1_1(feat5_sar)
        feat4_sar = self.shortcut_conv2_1(feat4_sar)
        feat3_sar = self.shortcut_conv3_1(feat3_sar)
        feat2_sar = self.shortcut_conv4_1(feat2_sar)

        # MGFM Fusion
        fusion_feat1 = self.fusion_1(feat1_opt, feat2_sar)
        fusion_feat2 = self.fusion_2(feat2_opt, feat3_sar)
        fusion_feat3 = self.fusion_3(feat3_opt, feat4_sar)
        fusion_feat4 = self.fusion_4(feat4_opt, feat5_sar)

        # Decoder
        x0_h, x0_w = fusion_feat1.size(2), fusion_feat1.size(3)
        x1 = F.interpolate(fusion_feat2, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = F.interpolate(fusion_feat3, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x3 = F.interpolate(fusion_feat4, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x = torch.cat([fusion_feat1, x1, x2, x3], 1)
        x = self.last_layer(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)

        return x

    def freeze_backbone(self):
        for param in self.opt_encoder.parameters():
            param.requires_grad = False
        # for param in self.sar_encoder.parameters():
        #     param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.opt_encoder.parameters():
            param.requires_grad = True
        # for param in self.sar_encoder.parameters():
        #     param.requires_grad = True

if __name__ == "__main__":
    model = MGFNet(num_classes=8,pretrained=True)
    model.train()
    sar = torch.randn(2, 1, 256, 256)
    opt = torch.randn(2, 3, 256, 256)
    # print(model)
    print("input:", sar.shape, opt.shape)
    print("output:", model(opt, sar).shape)
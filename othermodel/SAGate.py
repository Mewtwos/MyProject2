import time
from collections import OrderedDict
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F, Parameter

bn_eps = 1e-5
bn_momentum = 0.1


class FilterLayer(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16):
        super(FilterLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, out_planes // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // reduction, out_planes),
            nn.Sigmoid()
        )
        self.out_planes = out_planes

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, self.out_planes, 1, 1)
        return y


class FSP(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16):
        super(FSP, self).__init__()
        self.filter = FilterLayer(2 * in_planes, out_planes, reduction)

    def forward(self, guidePath, mainPath):
        combined = torch.cat((guidePath, mainPath), dim=1)
        channel_weight = self.filter(combined)
        out = mainPath + channel_weight * guidePath
        return out


class SAGate(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16, bn_momentum=0.0003):
        self.init__ = super(SAGate, self).__init__()
        self.in_planes = in_planes
        self.bn_momentum = bn_momentum

        self.fsp_rgb = FSP(in_planes, out_planes, reduction)
        self.fsp_hha = FSP(in_planes, out_planes, reduction)

        self.gate_rgb = nn.Conv2d(in_planes * 2, 1, kernel_size=1, bias=True)
        self.gate_hha = nn.Conv2d(in_planes * 2, 1, kernel_size=1, bias=True)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        rgb, hha = x
        b, c, h, w = rgb.size()

        rec_rgb = self.fsp_rgb(hha, rgb)
        rec_hha = self.fsp_hha(rgb, hha)

        cat_fea = torch.cat([rec_rgb, rec_hha], dim=1)

        attention_vector_l = self.gate_rgb(cat_fea)
        attention_vector_r = self.gate_hha(cat_fea)

        attention_vector = torch.cat([attention_vector_l, attention_vector_r], dim=1)
        attention_vector = self.softmax(attention_vector)
        attention_vector_l, attention_vector_r = attention_vector[:, 0:1, :, :], attention_vector[:, 1:2, :, :]
        merge_feature = rgb * attention_vector_l + hha * attention_vector_r

        rgb_out = (rgb + merge_feature) / 2
        hha_out = (hha + merge_feature) / 2

        rgb_out = self.relu1(rgb_out)
        hha_out = self.relu2(hha_out)

        return [rgb_out, hha_out], merge_feature


class DualResNet(nn.Module):

    def __init__(self, block, layers, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 bn_momentum=0.1, deep_stem=False, stem_width=32, inplace=True):
        self.inplanes = stem_width * 2 if deep_stem else 64
        super(DualResNet, self).__init__()
        if deep_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, stem_width, kernel_size=3, stride=2, padding=1,
                          bias=False),
                norm_layer(stem_width, eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(inplace=inplace),
                nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1,
                          padding=1,
                          bias=False),
                norm_layer(stem_width, eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(inplace=inplace),
                nn.Conv2d(stem_width, stem_width * 2, kernel_size=3, stride=1,
                          padding=1,
                          bias=False),
            )
            self.hha_conv1 = nn.Sequential(
                nn.Conv2d(3, stem_width, kernel_size=3, stride=2, padding=1,
                          bias=False),
                norm_layer(stem_width, eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(inplace=inplace),
                nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1,
                          padding=1,
                          bias=False),
                norm_layer(stem_width, eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(inplace=inplace),
                nn.Conv2d(stem_width, stem_width * 2, kernel_size=3, stride=1,
                          padding=1,
                          bias=False),
            )
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
            self.hha_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                       bias=False)

        self.bn1 = norm_layer(stem_width * 2 if deep_stem else 64, eps=bn_eps,
                              momentum=bn_momentum)
        self.hha_bn1 = norm_layer(stem_width * 2 if deep_stem else 64, eps=bn_eps,
                                  momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.hha_relu = nn.ReLU(inplace=inplace)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.hha_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, norm_layer, 64, layers[0],
                                       inplace,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer2 = self._make_layer(block, norm_layer, 128, layers[1],
                                       inplace, stride=2,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer3 = self._make_layer(block, norm_layer, 256, layers[2],
                                       inplace, stride=2,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer4 = self._make_layer(block, norm_layer, 512, layers[3],
                                       inplace, stride=2,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)

        self.sagates = nn.ModuleList([
            SAGate(in_planes=256, out_planes=256, bn_momentum=bn_momentum),
            SAGate(in_planes=512, out_planes=512, bn_momentum=bn_momentum),
            SAGate(in_planes=1024, out_planes=1024, bn_momentum=bn_momentum),
            SAGate(in_planes=2048, out_planes=2048, bn_momentum=bn_momentum)
        ])

    def _make_layer(self, block, norm_layer, planes, blocks, inplace=True,
                    stride=1, bn_eps=1e-5, bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion, eps=bn_eps,
                           momentum=bn_momentum),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, norm_layer, bn_eps,
                            bn_momentum, downsample, inplace))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                norm_layer=norm_layer, bn_eps=bn_eps,
                                bn_momentum=bn_momentum, inplace=inplace))

        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)

        x2 = self.hha_conv1(x2)
        x2 = self.hha_bn1(x2)
        x2 = self.hha_relu(x2)
        x2 = self.hha_maxpool(x2)

        x = [x1, x2]
        blocks = []
        merges = []
        x = self.layer1(x)
        x, merge = self.sagates[0](x)
        blocks.append(x)
        merges.append(merge)

        x = self.layer2(x)
        x, merge = self.sagates[1](x)
        blocks.append(x)
        merges.append(merge)

        x = self.layer3(x)
        x, merge = self.sagates[2](x)
        blocks.append(x)
        merges.append(merge)

        x = self.layer4(x)
        x, merge = self.sagates[3](x)
        blocks.append(x)
        merges.append(merge)

        return blocks, merges


class DualBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,
                 norm_layer=None, bn_eps=1e-5, bn_momentum=0.1,
                 downsample=None, inplace=True):
        super(DualBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = norm_layer(planes * self.expansion, eps=bn_eps,
                              momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample

        self.hha_conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.hha_bn1 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.hha_conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False)
        self.hha_bn2 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.hha_conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                                   bias=False)
        self.hha_bn3 = norm_layer(planes * self.expansion, eps=bn_eps,
                                  momentum=bn_momentum)
        self.hha_relu = nn.ReLU(inplace=inplace)
        self.hha_relu_inplace = nn.ReLU(inplace=True)
        self.hha_downsample = downsample

        self.stride = stride
        self.inplace = inplace

    def forward(self, x):
        # first path
        x1 = x[0]
        residual1 = x1

        out1 = self.conv1(x1)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)

        out1 = self.conv2(out1)
        out1 = self.bn2(out1)
        out1 = self.relu(out1)

        out1 = self.conv3(out1)
        out1 = self.bn3(out1)

        if self.downsample is not None:
            residual1 = self.downsample(x1)

        # second path
        x2 = x[1]
        residual2 = x2

        out2 = self.hha_conv1(x2)
        out2 = self.hha_bn1(out2)
        out2 = self.hha_relu(out2)

        out2 = self.hha_conv2(out2)
        out2 = self.hha_bn2(out2)
        out2 = self.hha_relu(out2)

        out2 = self.hha_conv3(out2)
        out2 = self.hha_bn3(out2)

        if self.hha_downsample is not None:
            residual2 = self.hha_downsample(x2)

        out1 += residual1
        out2 += residual2
        out1 = self.relu_inplace(out1)
        out2 = self.relu_inplace(out2)

        return [out1, out2]


def load_dualpath_model(model, model_file, is_restore=False):
    # load raw state_dict
    t_start = time.time()
    if isinstance(model_file, str):
        raw_state_dict = torch.load(model_file)

        if 'model' in raw_state_dict.keys():
            raw_state_dict = raw_state_dict['model']
    else:
        raw_state_dict = model_file
    # copy to  hha backbone
    state_dict = {}
    for k, v in raw_state_dict.items():
        state_dict[k.replace('.bn.', '.')] = v
        if k.find('conv1') >= 0:
            state_dict[k] = v
            state_dict[k.replace('conv1', 'hha_conv1')] = v
        if k.find('conv2') >= 0:
            state_dict[k] = v
            state_dict[k.replace('conv2', 'hha_conv2')] = v
        if k.find('conv3') >= 0:
            state_dict[k] = v
            state_dict[k.replace('conv3', 'hha_conv3')] = v
        if k.find('bn1') >= 0:
            state_dict[k] = v
            state_dict[k.replace('bn1', 'hha_bn1')] = v
        if k.find('bn2') >= 0:
            state_dict[k] = v
            state_dict[k.replace('bn2', 'hha_bn2')] = v
        if k.find('bn3') >= 0:
            state_dict[k] = v
            state_dict[k.replace('bn3', 'hha_bn3')] = v
        if k.find('downsample') >= 0:
            state_dict[k] = v
            state_dict[k.replace('downsample', 'hha_downsample')] = v
    t_ioend = time.time()

    if is_restore:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=False)

    del state_dict
    # t_end = time.time()
    # logger.info(
    #     "Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
    #         t_ioend - t_start, t_end - t_ioend))

    return model


def resnet101(pretrained_model=None, **kwargs):
    model = DualResNet(DualBottleneck, [3, 4, 23, 3], **kwargs)

    if pretrained_model is not None:
        model = load_dualpath_model(model, pretrained_model)
    return model


class ASPP(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation_rates=(12, 24, 36),
                 hidden_channels=256,
                 norm_act=nn.BatchNorm2d,
                 pooling_size=None):
        super(ASPP, self).__init__()
        self.pooling_size = pooling_size

        self.map_convs = nn.ModuleList([
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[0],
                      padding=dilation_rates[0]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[1],
                      padding=dilation_rates[1]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[2],
                      padding=dilation_rates[2])
        ])
        self.map_bn = norm_act(hidden_channels * 4)

        self.global_pooling_conv = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.global_pooling_bn = norm_act(hidden_channels)

        self.red_conv = nn.Conv2d(hidden_channels * 4, out_channels, 1, bias=False)
        self.pool_red_conv = nn.Conv2d(hidden_channels, out_channels, 1, bias=False)
        self.red_bn = norm_act(out_channels)

        self.leak_relu = nn.LeakyReLU()

    def forward(self, x):
        # Map convolutions
        out = torch.cat([m(x) for m in self.map_convs], dim=1)
        out = self.map_bn(out)
        out = self.leak_relu(out)  # add activation layer
        out = self.red_conv(out)

        # Global pooling
        pool = self._global_pooling(x)
        pool = self.global_pooling_conv(pool)
        pool = self.global_pooling_bn(pool)

        pool = self.leak_relu(pool)  # add activation layer

        pool = self.pool_red_conv(pool)
        if self.training or self.pooling_size is None:
            pool = pool.repeat(1, 1, x.size(2), x.size(3))

        out += pool
        out = self.red_bn(out)
        out = self.leak_relu(out)  # add activation layer
        return out

    def _global_pooling(self, x):
        if self.training or self.pooling_size is None:
            pool = x.view(x.size(0), x.size(1), -1).mean(dim=-1)
            pool = pool.view(x.size(0), x.size(1), 1, 1)
        else:
            assert False
            # pooling_size = (min(try_index(self.pooling_size, 0), x.shape[2]),
            #                 min(try_index(self.pooling_size, 1), x.shape[3]))
            # padding = (
            #     (pooling_size[1] - 1) // 2,
            #     (pooling_size[1] - 1) // 2 if pooling_size[1] % 2 == 1 else (pooling_size[1] - 1) // 2 + 1,
            #     (pooling_size[0] - 1) // 2,
            #     (pooling_size[0] - 1) // 2 if pooling_size[0] % 2 == 1 else (pooling_size[0] - 1) // 2 + 1
            # )
            #
            # pool = nn.functional.avg_pool2d(x, pooling_size, stride=1)
            # pool = nn.functional.pad(pool, pad=padding, mode="replicate")
        return pool


class _FCNHead(nn.Module):
    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d, bn_momentum=0.003):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )

    def forward(self, x):
        return self.block(x)


class Head(nn.Module):
    def __init__(self, classify_classes, norm_act=nn.BatchNorm2d, bn_momentum=0.0003):
        super(Head, self).__init__()

        self.classify_classes = classify_classes
        self.aspp = ASPP(2048, 256, [6, 12, 18], norm_act=norm_act)

        self.reduce = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            norm_act(48, momentum=bn_momentum),
            nn.ReLU(),
        )

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_act(256, momentum=bn_momentum),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_act(256, momentum=bn_momentum),
                                       nn.ReLU(),
                                       nn.Dropout2d(0.1)
                                       )

        self.classify = nn.Conv2d(in_channels=256, out_channels=self.classify_classes, kernel_size=1,
                                  stride=1, padding=0, dilation=1, bias=True)

        self.auxlayer = _FCNHead(2048, classify_classes, bn_momentum=bn_momentum, norm_layer=norm_act)

    def forward(self, f_list):
        f = f_list[-1]
        encoder_out = f
        f = self.aspp(f)

        low_level_features = f_list[0]
        low_h, low_w = low_level_features.size(2), low_level_features.size(3)
        low_level_features = self.reduce(low_level_features)

        f = F.interpolate(f, size=(low_h, low_w), mode='bilinear', align_corners=True)
        f = torch.cat((f, low_level_features), dim=1)
        f = self.last_conv(f)

        pred = self.classify(f)

        aux_fm = self.auxlayer(encoder_out)
        return pred, aux_fm


class DeepLab(nn.Module):
    def __init__(self, out_planes, norm_layer, pretrained_model=None):
        super(DeepLab, self).__init__()
        self.backbone = resnet101(pretrained_model, norm_layer=norm_layer,
                                  bn_eps=bn_eps,
                                  bn_momentum=bn_momentum,
                                  deep_stem=True, stem_width=64)
        self.dilate = 2

        for m in self.backbone.layer4.children():
            m.apply(partial(self._nostride_dilate, dilate=self.dilate))
            self.dilate *= 2

        self.head = Head(out_planes, norm_layer, bn_momentum)

        self.business_layer = []
        self.business_layer.append(self.head)

    def forward(self, data, hha):
        hha = hha.unsqueeze(1).repeat(1, 3, 1, 1)
        b, c, h, w = data.shape
        blocks, merges = self.backbone(data, hha)
        pred, aux_fm = self.head(merges)
        pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)
        aux_fm = F.interpolate(aux_fm, size=(h, w), mode='bilinear', align_corners=True)

        # if label is not None:  # training
        #     loss = self.criterion(pred, label)
        #     loss_aux = self.criterion(aux_fm, label)

        #     return loss, loss_aux

        return pred

    # @staticmethod
    def _nostride_dilate(self, m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)


def _ntuple(n):
    def parse(x):
        if isinstance(x, list) or isinstance(x, tuple):
            return x
        return tuple([x] * n)

    return parse


_pair = _ntuple(2)


class Conv2_5D_Depth(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True,
                 pixel_size=1):
        super(Conv2_5D_Depth, self).__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel_size_prod = self.kernel_size[0] * self.kernel_size[1]
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pixel_size = pixel_size
        assert self.kernel_size_prod % 2 == 1

        self.weight_0 = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        self.weight_1 = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        self.weight_2 = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x, depth, camera_params):
        N, C, H, W = x.size(0), x.size(1), x.size(2), x.size(3)
        out_H = (H + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        out_W = (W + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1

        intrinsic = camera_params['intrinsic']
        x_col = F.unfold(x, self.kernel_size, dilation=self.dilation, padding=self.padding,
                         stride=self.stride)  # N*(C*kh*kw)*(out_H*out_W)
        x_col = x_col.view(N, C, self.kernel_size_prod, out_H * out_W)
        depth_col = F.unfold(depth, self.kernel_size, dilation=self.dilation, padding=self.padding,
                             stride=self.stride)  # N*(kh*kw)*(out_H*out_W)
        valid_mask = 1 - depth_col.eq(0.).to(torch.float32)

        valid_mask = valid_mask * valid_mask[:, self.kernel_size_prod // 2, :].view(N, 1, out_H * out_W)
        depth_col *= valid_mask
        valid_mask = valid_mask.view(N, 1, self.kernel_size_prod, out_H * out_W)

        center_depth = depth_col[:, self.kernel_size_prod // 2, :].view(N, 1, out_H * out_W)
        # grid_range = self.pixel_size * center_depth / (intrinsic['fx'].view(N,1,1) * camera_params['scale'].view(N,1,1))
        grid_range = self.pixel_size * self.dilation[0] * center_depth / intrinsic['fx'].view(N, 1, 1)

        mask_0 = torch.abs(depth_col - (center_depth + grid_range)).le(grid_range / 2).view(N, 1, self.kernel_size_prod,
                                                                                            out_H * out_W).to(
            torch.float32)
        mask_1 = torch.abs(depth_col - (center_depth)).le(grid_range / 2).view(N, 1, self.kernel_size_prod,
                                                                               out_H * out_W).to(torch.float32)
        mask_1 = (mask_1 + 1 - valid_mask).clamp(min=0., max=1.)
        mask_2 = torch.abs(depth_col - (center_depth - grid_range)).le(grid_range / 2).view(N, 1, self.kernel_size_prod,
                                                                                            out_H * out_W).to(
            torch.float32)
        output = torch.matmul(self.weight_0.view(-1, C * self.kernel_size_prod),
                              (x_col * mask_0).view(N, C * self.kernel_size_prod, out_H * out_W))
        output += torch.matmul(self.weight_1.view(-1, C * self.kernel_size_prod),
                               (x_col * mask_1).view(N, C * self.kernel_size_prod, out_H * out_W))
        output += torch.matmul(self.weight_2.view(-1, C * self.kernel_size_prod),
                               (x_col * mask_2).view(N, C * self.kernel_size_prod, out_H * out_W))
        output = output.view(N, -1, out_H, out_W)
        if self.bias:
            output += self.bias.view(1, -1, 1, 1)
        return output

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class Malleable_Conv2_5D_Depth(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, pixel_size=1,
                 anchor_init=[-2., -1., 0., 1., 2.], scale_const=100, fix_center=False, adjust_to_scale=False):
        super(Malleable_Conv2_5D_Depth, self).__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel_size_prod = self.kernel_size[0] * self.kernel_size[1]
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pixel_size = pixel_size
        self.fix_center = fix_center
        self.adjust_to_scale = adjust_to_scale
        assert self.kernel_size_prod % 2 == 1

        self.weight_0 = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        self.weight_1 = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        self.weight_2 = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        self.depth_anchor = Parameter(torch.tensor(anchor_init, requires_grad=True).view(1, 5, 1, 1))
        # self.depth_bias = Parameter(torch.tensor([0.,0.,0.,0.,0.], requires_grad=True).view(1,5,1,1))
        self.temperature = Parameter(torch.tensor([1.], requires_grad=True))
        self.kernel_weight = Parameter(torch.tensor([0., 0., 0.], requires_grad=True))
        self.scale_const = scale_const
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x, depth, camera_params):
        N, C, H, W = x.size(0), x.size(1), x.size(2), x.size(3)
        out_H = (H + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        out_W = (W + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1

        intrinsic = camera_params['intrinsic']
        x_col = F.unfold(x, self.kernel_size, dilation=self.dilation, padding=self.padding,
                         stride=self.stride)  # N*(C*kh*kw)*(out_H*out_W)
        x_col = x_col.view(N, C, self.kernel_size_prod, out_H * out_W)
        depth_col = F.unfold(depth, self.kernel_size, dilation=self.dilation, padding=self.padding,
                             stride=self.stride)  # N*(kh*kw)*(out_H*out_W)
        valid_mask = 1 - depth_col.eq(0.).to(torch.float32)

        valid_mask = valid_mask * valid_mask[:, self.kernel_size_prod // 2, :].view(N, 1, out_H * out_W)
        depth_col *= valid_mask
        valid_mask = valid_mask.view(N, 1, self.kernel_size_prod, out_H * out_W)

        center_depth = depth_col[:, self.kernel_size_prod // 2, :].view(N, 1, out_H * out_W)
        if self.adjust_to_scale:
            grid_range = self.pixel_size * self.dilation[0] * center_depth / (
                    intrinsic['fx'].view(N, 1, 1) * camera_params['scale'].view(N, 1, 1))
        else:
            grid_range = self.pixel_size * self.dilation[0] * center_depth / intrinsic['fx'].view(N, 1, 1)
        depth_diff = (depth_col - center_depth).view(N, 1, self.kernel_size_prod,
                                                     out_H * out_W)  # N*1*(kh*kw)*(out_H*out_W)
        relative_diff = depth_diff * self.scale_const / (
                1e-5 + grid_range.view(N, 1, 1, out_H * out_W) * self.scale_const)
        depth_logit = -(((relative_diff - self.depth_anchor).pow(2)) / (
                1e-5 + torch.clamp(self.temperature, min=0.)))  # N*5*(kh*kw)*(out_H*out_W)
        if self.fix_center:
            depth_logit[:, 2, :, :] = -(
                    ((relative_diff - 0.).pow(2)) / (1e-5 + torch.clamp(self.temperature, min=0.))).view(N,
                                                                                                         self.kernel_size_prod,
                                                                                                         out_H * out_W)

        depth_out_range_0 = (depth_diff < self.depth_anchor[0, 0, 0, 0]).to(torch.float32).view(N,
                                                                                                self.kernel_size_prod,
                                                                                                out_H * out_W)
        depth_out_range_4 = (depth_diff > self.depth_anchor[0, 4, 0, 0]).to(torch.float32).view(N,
                                                                                                self.kernel_size_prod,
                                                                                                out_H * out_W)
        depth_logit[:, 0, :, :] = depth_logit[:, 0, :, :] * (1 - 2 * depth_out_range_0)
        depth_logit[:, 4, :, :] = depth_logit[:, 4, :, :] * (1 - 2 * depth_out_range_4)

        depth_class = F.softmax(depth_logit, dim=1)  # N*5*(kh*kw)*(out_H*out_W)

        mask_0 = depth_class[:, 1, :, :].view(N, 1, self.kernel_size_prod, out_H * out_W).to(torch.float32)
        mask_1 = depth_class[:, 2, :, :].view(N, 1, self.kernel_size_prod, out_H * out_W).to(torch.float32)
        mask_2 = depth_class[:, 3, :, :].view(N, 1, self.kernel_size_prod, out_H * out_W).to(torch.float32)

        invalid_mask_bool = valid_mask.eq(0.)

        mask_0 = mask_0 * valid_mask
        mask_1 = mask_1 * valid_mask
        mask_2 = mask_2 * valid_mask
        mask_0[invalid_mask_bool] = 1. / 5.
        mask_1[invalid_mask_bool] = 1. / 5.
        mask_2[invalid_mask_bool] = 1. / 5.

        weight = F.softmax(self.kernel_weight, dim=0) * 3  # ???
        output = torch.matmul(self.weight_0.view(-1, C * self.kernel_size_prod),
                              (x_col * mask_0).view(N, C * self.kernel_size_prod, out_H * out_W)) * weight[0]
        output += torch.matmul(self.weight_1.view(-1, C * self.kernel_size_prod),
                               (x_col * mask_1).view(N, C * self.kernel_size_prod, out_H * out_W)) * weight[1]
        output += torch.matmul(self.weight_2.view(-1, C * self.kernel_size_prod),
                               (x_col * mask_2).view(N, C * self.kernel_size_prod, out_H * out_W)) * weight[2]
        output = output.view(N, -1, out_H, out_W)
        if self.bias:
            output += self.bias.view(1, -1, 1, 1)
        return output

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                  **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, Conv2_5D_Depth) or isinstance(m, Malleable_Conv2_5D_Depth):
            conv_init(m.weight_0, **kwargs)
            conv_init(m.weight_1, **kwargs)
            conv_init(m.weight_2, **kwargs)
        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                          **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                      **kwargs)


if __name__ == "__main__":
    pretrained_model = '/home/lvhaitao/resnet101_v1c.pth'
    model = DeepLab(6,
                    pretrained_model=pretrained_model,
                    norm_layer=nn.BatchNorm2d).cuda()
    init_weight(model.business_layer, nn.init.kaiming_normal_,
                nn.BatchNorm2d, 1e-5, 0.1,
                mode='fan_in', nonlinearity='relu')
    x = torch.randn(10, 3, 256, 256).cuda()
    hha = torch.randn(10, 256, 256).cuda()
    y = model(x, hha)
    print(y.size())


import os
import torch
from torch import nn
from thop import profile
from thop import clever_format
from convnextv2 import convnextv2_unet_modify2
from othermodel.ESANet import ESANet
from othermodel.CMFNet import CMFNet
# from othermodel.rs3mamba import RS3Mamba
from othermodel.MAResUNet import MAResUNet
from othermodel.ABCNet import ABCNet
from othermodel.unetformer import UNetFormer
from othermodel.RFNet import RFNet, resnet18
from othermodel.SAGate import DeepLab, init_weight
from othermodel.ACNet import ACNet
from othermodel.CMGFNet import FuseNet

# 创建模型实例
net = convnextv2_unet_modify2.__dict__["convnextv2_unet_tiny"](
            num_classes=6,
            drop_path_rate=0.1,
            head_init_scale=0.001,
            patch_size=16,  
            use_orig_stem=False,
            in_chans=3,
        ).cuda()

# net = ESANet(
#     height=256,
#     width=256,
#     num_classes=6,
#     pretrained_on_imagenet=True,
#     pretrained_dir="/home/lvhaitao/pretrained_model",
#     encoder_rgb="resnet34",
#     encoder_depth="resnet34",
#     encoder_block="NonBottleneck1D",
#     nr_decoder_blocks=[3, 3, 3],
#     channels_decoder=[512, 256, 128],
#     upsampling="learned-3x3-zeropad"
# )

# net = CMFNet()

# net = RS3Mamba(num_classes=6).cuda()

# net = MAResUNet(num_classes=6).cuda()

# net = ABCNet(6).cuda()

# net = UNetFormer(num_classes=6, pretrained=False).cuda()

# resnet = resnet18(pretrained=True, efficient=False, use_bn=True)
# net = RFNet(resnet, num_classes=6, use_bn=True).cuda()

# net = DeepLab(6, pretrained_model=None, norm_layer=nn.BatchNorm2d).cuda()

# net = ACNet(num_class=6, pretrained=False).cuda()

# net = FuseNet(num_classes=6, pretrained=False).cuda()

net.eval()
# 创建两个随机输入张量
input1 = torch.randn(1, 3, 256, 256).cuda()
input2 = torch.randn(1, 256, 256).cuda()

# 使用thop分析模型的运算量和参数量
flops, params = profile(net, inputs=(input1, input2))

# 将结果转换为更易于阅读的格式
flops, params = clever_format([flops, params], '%.3f')

print(f"运算量：{flops}, 参数量：{params}")


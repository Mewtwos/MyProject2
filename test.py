from ptflops import get_model_complexity_info
import torch
import torchvision.models as models

from convnextv2 import convnextv2_unet_modify3

# 创建一个模型实例，这里以 ResNet-50 为例
net = convnextv2_unet_modify3.__dict__["convnextv2_unet_tiny"](
            num_classes=6,
            drop_path_rate=0.1,
            patch_size=16,  
            use_orig_stem=False,
            in_chans=3,
        ).cuda()

# 输入模型的尺寸，这个尺寸应该与模型输入的尺寸相匹配
input_res = (4, 256, 256)

# 使用 ptflops 计算模型的 FLOPs 和参数量
macs, params = get_model_complexity_info(net, input_res, as_strings=True, print_per_layer_stat=True)

print(f"模型 FLOPs: {macs}")
print(f"模型参数量: {params}")

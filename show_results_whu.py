from PIL import Image
# from matplotlib import pyplot as plt
import numpy as np
import torch
from othermodel.ABCNet import ABCNet
from skimage import io
from othermodel.CMFNet import CMFNet
from othermodel.CMGFNet import FuseNet
from othermodel.MAResUNet import MAResUNet
# from othermodel.Transunet import CONFIGS as CONFIGS_ViT_seg
from othermodel.Transunet import VisionTransformer as TransUNet
from othermodel.ukan import UKAN
from othermodel.unetformer import UNetFormer
# from othermodel.rs3mamba import RS3Mamba
from convnextv2 import convnextv2_unet_modify2, convnextv2_unet_modify3
from othermodel.ACNet import ACNet
from othermodel.RFNet import RFNet, resnet18
from othermodel.ESANet import ESANet
from othermodel.SAGate import DeepLab, init_weight
import torch.nn as nn
from model.vitcross_seg_modeling import VisionTransformer as ViT_seg
from model.vitcross_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from torchvision import transforms


transform = transforms.Compose([
    transforms.ToTensor(),
    ])

datas = []
sars = []
index = ["NH49E002012_161.tif", "NH49E002016_204.tif"]
data_dir = "/data/lvhaitao/dataset/whu-opt-sar/test/opt/{}"
sar_dir = "/data/lvhaitao/dataset/whu-opt-sar/test/sar/{}"

for i in range(2):
    data = Image.open(data_dir.format(index[i]))
    sar = Image.open(sar_dir.format(index[i]))
    data = transform(data)[:3,:,:]
    sar = transform(sar)
    datas.append(data)
    sars.append(sar)

# net = ABCNet(8).cuda()
# net.load_state_dict(torch.load("/home/lvhaitao/MyProject2/testsavemodel/ABCNet_whu_epoch27_80.6287467074232"))

# net = CMFNet(out_channels=8).cuda()
# net.load_state_dict(torch.load("/home/lvhaitao/MyProject2/testsavemodel/CMFNet_whu_epoch26_82.3170551637403"))

#CMGFNet
# net = FuseNet(num_classes=8, pretrained=False).cuda()
# net.load_state_dict(torch.load("/home/lvhaitao/MyProject2/testsavemodel/CMGFNet_whu_epoch12_81.28371906929276"))

# net = MAResUNet(num_classes=8).cuda()
# net.load_state_dict(torch.load("/home/lvhaitao/MyProject2/testsavemodel/maresunet_whu_epoch26_80.83489190964472"))

# net = UNetFormer(num_classes=8,pretrained=False).cuda()
# net.load_state_dict(torch.load("/home/lvhaitao/MyProject2/testsavemodel/unetformer_whu_epoch14_81.03740536436743"))

net = convnextv2_unet_modify3.__dict__["convnextv2_unet_tiny"](
            num_classes=8,
            drop_path_rate=0.1,
            patch_size=16,  ###原来是16
            use_orig_stem=False,
            in_chans=3,
        ).cuda()
# net.load_state_dict(torch.load("/home/lvhaitao/MyProject2/testsavemodel/mffnet_whu_epoch13_81.83644067673457"))
net.load_state_dict(torch.load("/home/lvhaitao/finetune/mffnet_whu_finetune_3"))

# net = RS3Mamba(num_classes=8).cuda()
# net.load_state_dict(torch.load("/home/lvhaitao/MyProject2/testsavemodel/rs3mamba_whu_epoch26_81.17447989327567"))

# net = ACNet(num_class=8, pretrained=False).cuda()
# net.load_state_dict(torch.load("/home/lvhaitao/MyProject2/testsavemodel/ACNet_whu_epoch26_81.93361970032154"))

# resnet = resnet18(pretrained=False, efficient=False, use_bn=True)
# net = RFNet(resnet, num_classes=8, use_bn=True).cuda()
# net.load_state_dict(torch.load("/home/lvhaitao/MyProject2/testsavemodel/rfnet_whu_epoch12_81.23649026260895"))

# net = ESANet(
#     height=256,
#     width=256,
#     num_classes=8,
#     pretrained_on_imagenet=True,
#     pretrained_dir="/home/lvhaitao/pretrained_model",
#     encoder_rgb="resnet34",
#     encoder_depth="resnet34",
#     encoder_block="NonBottleneck1D",
#     nr_decoder_blocks=[3, 3, 3],
#     channels_decoder=[512, 256, 128],
#     upsampling="learned-3x3-zeropad"
# ).cuda()
# net.load_state_dict(torch.load("/home/lvhaitao/MyProject2/testsavemodel/ESANet_whu_epoch26_81.92489027166043"))

#SAGate
# pretrained_model = '/home/lvhaitao/resnet101_v1c.pth'
# net = DeepLab(8, pretrained_model=pretrained_model, norm_layer=nn.BatchNorm2d).cuda()
# init_weight(net.business_layer, nn.init.kaiming_normal_,nn.BatchNorm2d, 1e-5, 0.1,mode='fan_in', nonlinearity='relu')
# net.load_state_dict(torch.load("/home/lvhaitao/MyProject2/testsavemodel/sagate_whu_epoch12_82.76008424304781"))


net.eval()

def decode_segmentation(output, palette):
    # 获取每个像素的类别索引，形状为 [b, 256, 256]
    output_class = torch.argmax(output, dim=1)  # 选择每个像素的最大类别

    # 创建一个空的 RGB 图像，形状为 [b, 256, 256, 3]
    decoded_images = torch.zeros((output_class.size(0), output_class.size(1), output_class.size(2), 3), dtype=torch.uint8)

    # 将每个类别的颜色填充到图像中
    for class_idx, color in palette.items():
        mask = (output_class == class_idx)
        decoded_images[mask] = torch.tensor(color, dtype=torch.uint8)

    return decoded_images

palette = {
    0: [0, 0, 0],  # 类别0对应黑色 backgroud
    1: [204, 102, 0],  # 类别1对应棕色 farmland
    2: [255, 0, 0],  # 类别2对应红色  city
    3: [255, 255, 0],  # 类别3对应黄色 village
    4: [0, 0, 255],  # 类别4对应蓝色  water
    5: [85, 167, 0],  # 类别5对应绿色   forest
    6: [0, 255, 255],  # 类别6对应靛蓝色  road
    7: [153, 102, 153]  # 类别7对应紫色  others
}

for i in range(2):
    with torch.no_grad():
        data = datas[i].unsqueeze(0).cuda()
        sar = sars[i].unsqueeze(0).cuda()
        output = net(data, sar.squeeze(0))
    decoded_output = decode_segmentation(output, palette)
    #转为numpy
    decoded_output = decoded_output.squeeze().cpu().numpy().astype(np.uint8)
    image = Image.fromarray(decoded_output)
    image.save("/home/lvhaitao/whu_label{}.png".format(i+1))

print("结束")


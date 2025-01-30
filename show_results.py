from PIL import Image
# from matplotlib import pyplot as plt
import numpy as np
import torch
from othermodel.ABCNet import ABCNet
from skimage import io
from othermodel.CMFNet import CMFNet
from othermodel.CMGFNet import FuseNet
from othermodel.MAResUNet import MAResUNet
from othermodel.Transunet import CONFIGS as CONFIGS_ViT_seg
from othermodel.Transunet import VisionTransformer as TransUNet
from othermodel.ukan import UKAN
from othermodel.unetformer import UNetFormer
# from othermodel.rs3mamba import RS3Mamba, load_pretrained_ckpt
from convnextv2 import convnextv2_unet_modify2
from othermodel.ACNet import ACNet
from othermodel.RFNet import RFNet, resnet18
from othermodel.ESANet import ESANet
from othermodel.SAGate import DeepLab, init_weight
import torch.nn as nn


vaihingen_data = {}
vaihingen_dsm = {}
patch_size = 256
dataset = "Vaihingen"
# dataset = "Potsdam"

if dataset == "Vaihingen":
    index_dict = {"1":30, "2":30, "3":30, "4":30}
    x_dict = {"1":2096, "2":1530, "3":510, "4":877}
    y_dict = {"1":1441, "2":544, "3":1601, "4":1244}
    dataset_dir = "/data/lvhaitao/dataset/Vaihingen/"
else:
    index_dict = {"1":"4_10", "2":"3_10", "3":"3_10", "4":"3_10"}
    x_dict = {"1":3226, "2":853, "3":3406, "4":4747}
    y_dict = {"1":3413, "2":5138, "3":2800, "4":2231}
    dataset_dir = "/data/lvhaitao/dataset/Potsdam/"

for i in range(4):
    x1 = x_dict[str(i+1)]
    y1 = y_dict[str(i+1)]
    x2 = x1 + patch_size
    y2 = y1 + patch_size
    index = index_dict[str(i+1)]
    if dataset == "Vaihingen":
        data = io.imread(dataset_dir+'top/top_mosaic_09cm_area{}.tif'.format(index))
        data = 1 / 255 * np.asarray(data.transpose((2, 0, 1)), dtype='float32')
        dsm = np.asarray(io.imread(dataset_dir+'dsm/dsm_09cm_matching_area{}.tif'.format(index)), dtype='float32')
    else:
        data = io.imread(dataset_dir+'4_Ortho_RGBIR/top_potsdam_{}_RGBIR.tif'.format(index))[:, :, :3].transpose((2, 0, 1))
        data = 1 / 255 * np.asarray(data, dtype='float32')
        dsm = np.asarray(io.imread(dataset_dir+'1_DSM_normalisation/dsm_potsdam_{}_normalized_lastools.jpg'.format(index)), dtype='float32')
    min = np.min(dsm)
    max = np.max(dsm)
    dsm = (dsm - min) / (max - min)

    data_p = data[:, x1:x2, y1:y2]
    dsm_p = dsm[x1:x2, y1:y2]
    vaihingen_data[str(i+1)] = data_p
    vaihingen_dsm[str(i+1)] = dsm_p


# net = ABCNet(6).cuda()
# net.load_state_dict(torch.load("E:/训练的模型/Vaingen/ABCNet_epoch34_91.37813089108155"))

# net = CMFNet().cuda()
# net.load_state_dict(torch.load("E:/训练的模型/Vaingen/CMFNet_epoch36_90.6976907313886"))

# net = FuseNet(num_classes=6, pretrained=False).cuda()
# net.load_state_dict(torch.load("E:/训练的模型/Vaingen/CMGFNet_epoch22_91.98291064860057"))

# net = MAResUNet(num_classes=6).cuda()
# net.load_state_dict(torch.load("/home/lvhaitao/MyProject2/savemodel/MAResUNet_epoch45_90.66164015953017"))
# net.load_state_dict(torch.load("E:/训练的模型/Vaingen/MARESUNET_epoch33_91.5188960086663"))

# config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
# config_vit.n_classes = 6
# config_vit.n_skip = 3
# config_vit.patches.grid = (14, 14)
# net = TransUNet(config_vit, 256, 6).cuda()
# net.load_state_dict(torch.load("/home/lvhaitao/MyProject2/savemodel/TransUNet_epoch47_90.94132738005878"))
# net.load_state_dict(torch.load("E:/训练的模型/Vaingen/transunet_epoch46_91.49426607288572"))

# net = UKAN(num_classes=6, input_h=256, input_w=256, deep_supervision=False, embed_dims=[128, 160, 256], no_kan=False).cuda()
# net.load_state_dict(torch.load("E:/训练的模型/Vaingen/UKAN_epoch47_90.2728950498298"))

# net = UNetFormer(num_classes=6,pretrained=False).cuda()
# net.load_state_dict(torch.load("/home/lvhaitao/unetformer_epoch50_91.6432166560365"))
# net.load_state_dict(torch.load("E:/训练的模型/Potsdam/unetformer_epoch44_90.46410366807537-potsdam"))

net = convnextv2_unet_modify2.__dict__["convnextv2_unet_tiny"](
            num_classes=6,
            drop_path_rate=0.1,
            head_init_scale=0.001,
            patch_size=16,  ###原来是16
            use_orig_stem=False,
            in_chans=3,
        ).cuda()
net.load_state_dict(torch.load("/home/lvhaitao/MyProject2/savemodel/MFFNet3_Vaihingen_epoch15_92.19754860296956"))
# net.load_state_dict(torch.load("/home/lvhaitao/MyProject2/savemodel/MFFNet2_Potsdam_epoch46_91.16632234306633"))

# net = RS3Mamba(num_classes=6).cuda()
# net.load_state_dict(torch.load("/home/lvhaitao/MyProject2/savemodel/RS3mamba_epoch45_90.393848321926"))

# net = ACNet(num_class=6, pretrained=False).cuda()
# net.load_state_dict(torch.load("/home/lvhaitao/MyProject2/savemodel/ACNet_Potsdam_epoch31_90.75246544602247"))

# resnet = resnet18(pretrained=False, efficient=False, use_bn=True)
# net = RFNet(resnet, num_classes=6, use_bn=True).cuda()
# net.load_state_dict(torch.load("/home/lvhaitao/MyProject2/savemodel/RFNet_Potsdam_epoch50_90.092240556577"))

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
# ).cuda()
# net.load_state_dict(torch.load("/home/lvhaitao/MyProject2/savemodel/ESANet_Potsdam_epoch48_90.59305027659295"))

#SAGate
# pretrained_model = '/home/lvhaitao/resnet101_v1c.pth'
# net = DeepLab(6, pretrained_model=pretrained_model, norm_layer=nn.BatchNorm2d).cuda()
# init_weight(net.business_layer, nn.init.kaiming_normal_,nn.BatchNorm2d, 1e-5, 0.1,mode='fan_in', nonlinearity='relu')
# net.load_state_dict(torch.load("/home/lvhaitao/MyProject2/savemodel/SAGate_Potsdam_epoch45_90.82541253543694"))

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

palette = {0 : (255, 255, 255), # Impervious surfaces (white)
           1 : (0, 0, 255),     # Buildings (blue)
           2 : (0, 255, 255),   # Low vegetation (cyan)
           3 : (0, 255, 0),     # Trees (green)
           4 : (255, 255, 0),   # Cars (yellow)
           5 : (255, 0, 0),     # Clutter (red)
        #    6 : (0, 0, 0)
           }       # Undefined (black)

for i in range(4):
    with torch.no_grad():
        data = torch.from_numpy(vaihingen_data[str(i+1)]).unsqueeze(0).cuda()
        dsm = torch.from_numpy(vaihingen_dsm[str(i+1)]).unsqueeze(0).cuda()
        output = net(data, dsm)
    decoded_output = decode_segmentation(output, palette)
    #转为numpy
    decoded_output = decoded_output.squeeze().cpu().numpy().astype(np.uint8)
    image = Image.fromarray(decoded_output)
    image.save("/home/lvhaitao/label{}.png".format(i+1))

print("结束")


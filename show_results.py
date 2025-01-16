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

dataset_dir = "/home/lvhaitao/Vaihingen_show"
# dataset_dir = "/home/lvhaitao/Potsdam_show"


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
# net.load_state_dict(torch.load("E:/训练的模型/Vaingen/unetformer_epoch50_91.6432166560365"))
# net.load_state_dict(torch.load("E:/训练的模型/Potsdam/unetformer_epoch44_90.46410366807537-potsdam"))

net = convnextv2_unet_modify2.__dict__["convnextv2_unet_tiny"](
            num_classes=6,
            drop_path_rate=0.1,
            head_init_scale=0.001,
            patch_size=16,  ###原来是16
            use_orig_stem=False,
            in_chans=3,
        ).cuda()
net.load_state_dict(torch.load("/home/lvhaitao/MyProject2/savemodel/MFFNet_Potsdam_epoch31_92.09799365583991"))

# net = RS3Mamba(num_classes=6).cuda()
# net.load_state_dict(torch.load("/home/lvhaitao/rs3mamba_epoch32_91.30714304907178-vaihingen"))

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
    0: (255, 255, 255),  # Impervious surfaces (white)
    1: (0, 0, 255),      # Buildings (blue)
    2: (0, 255, 255),    # Low vegetation (cyan)
    3: (0, 255, 0),      # Trees (green)
    4: (255, 255, 0),    # Cars (yellow)
    5: (255, 0, 0),      # Clutter (red)
    6: (0, 0, 0),        # Undefined (black)
}

for i in range(4):
    data = io.imread(dataset_dir + "/data{}.png".format(i+1))
    dsm = io.imread(dataset_dir + "/dsm{}.png".format(i+1))
    data = 1 / 255 * np.asarray(data.transpose((2, 0, 1)), dtype='float32')
    dsm = np.asarray(dsm, dtype='float32')
    min = np.min(dsm)
    max = np.max(dsm)
    dsm = (dsm - min) / (max - min)
    data = torch.from_numpy(data).unsqueeze(0).cuda()
    dsm = torch.from_numpy(dsm).unsqueeze(0).cuda()
    with torch.no_grad():
        output = net(data, dsm)
    decoded_output = decode_segmentation(output, palette)
    #转为numpy
    decoded_output = decoded_output.squeeze().cpu().numpy().astype(np.uint8)
    image = Image.fromarray(decoded_output)
    image.save("/home/lvhaitao/label{}.png".format(i+1))
    # plt.imshow(decoded_output[0].numpy())  # 显示第一个样本
    # plt.axis('off')  # 不显示坐标轴
    # plt.show()




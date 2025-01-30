import os
from PIL import Image
import numpy as np
import torch
from convnextv2 import convnextv2_unet_modify2
import torch.nn as nn
from skimage import io
from custom_repr import enable_custom_repr
enable_custom_repr()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.device_count.cache_clear() 

dataset_dir = "/data/lvhaitao/dataset/Potsdam/"
index = "4_10"
data = io.imread(dataset_dir+'4_Ortho_RGBIR/top_potsdam_{}_RGBIR.tif'.format(index))[:, :, :3].transpose((2, 0, 1))
data = 1 / 255 * np.asarray(data, dtype='float32')
dsm = np.asarray(io.imread(dataset_dir+'1_DSM_normalisation/dsm_potsdam_{}_normalized_lastools.jpg'.format(index)), dtype='float32')
min = np.min(dsm)
max = np.max(dsm)
dsm = (dsm - min) / (max - min)
data = torch.from_numpy(data).unsqueeze(0).cuda()
dsm = torch.from_numpy(dsm).unsqueeze(0).cuda()

net = convnextv2_unet_modify2.__dict__["convnextv2_unet_tiny"](
            num_classes=6,
            drop_path_rate=0.1,
            head_init_scale=0.001,
            patch_size=16,  ###原来是16
            use_orig_stem=False,
            in_chans=3,
        ).cuda()
# net.load_state_dict(torch.load("/home/lvhaitao/MyProject2/savemodel/MFFNet3_Vaihingen_epoch5_91.7385402774969"))
net.load_state_dict(torch.load("/home/lvhaitao/MyProject2/savemodel/MFFNet2_Potsdam_epoch46_91.16632234306633"))

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

with torch.no_grad():
    output = net(data, dsm)
decoded_output = decode_segmentation(output, palette)
#转为numpy
decoded_output = decoded_output.squeeze().cpu().numpy().astype(np.uint8)
image = Image.fromarray(decoded_output)
image.save("/home/lvhaitao/label_big.png")

print("结束")


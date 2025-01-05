from othermodel.unetformer import UNetFormer
import torch
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

net = UNetFormer(num_classes=6).cuda()
net.load_state_dict(torch.load('/home/lvhaitao/unetformer_epoch50_91.6432166560365'))
net.eval()

data = io.imread("/home/lvhaitao/data_p.png")
data = np.asarray(data.transpose((2, 0, 1)), dtype='float32') / 255.0
data = torch.from_numpy(data).cuda()
dsm = np.asarray(io.imread("/home/lvhaitao/dsm_p.png"), dtype='float32')
min = np.min(dsm)
max = np.max(dsm)
dsm = (dsm - min) / (max - min)
dsm = torch.from_numpy(dsm).cuda()
output = net(data.unsqueeze(0), dsm.unsqueeze(0).unsqueeze(0))

palette = {
    0: (255, 255, 255),  # Impervious surfaces (white)
    1: (0, 0, 255),      # Buildings (blue)
    2: (0, 255, 255),    # Low vegetation (cyan)
    3: (0, 255, 0),      # Trees (green)
    4: (255, 255, 0),    # Cars (yellow)
    5: (255, 0, 0),      # Clutter (red)
    6: (0, 0, 0),        # Undefined (black)
}

# def decode_segmentation(output, palette):
#     # 对 logits 进行 softmax 以获得每个类别的概率
#     output_softmax = F.softmax(output, dim=1)  # [b, 6, 256, 256]
    
#     # 获取每个像素的类别索引（最大概率的类别）
#     output_class = torch.argmax(output_softmax, dim=1)  # [b, 256, 256]

#     # 创建一个空的 RGB 图像，形状为 [b, 256, 256, 3]
#     decoded_images = torch.zeros((output_class.size(0), output_class.size(1), output_class.size(2), 3), dtype=torch.uint8)

#     # 将每个类别的颜色填充到图像中
#     for class_idx, color in palette.items():
#         mask = (output_class == class_idx)
#         decoded_images[mask] = torch.tensor(color, dtype=torch.uint8)

#     return decoded_images

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

# 解码输出
decoded_output = decode_segmentation(output, palette)

# 可视化第一张图像
plt.imshow(decoded_output[0].numpy())  # 显示第一个样本
plt.axis('off')  # 不显示坐标轴
plt.show()

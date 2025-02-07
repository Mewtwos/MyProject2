import os
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from convnextv2 import convnextv2_unet_modify2
from skimage import io
from othermodel.SAGate import DeepLab, init_weight


palette = {0 : (255, 255, 255), # Impervious surfaces (white)
           1 : (0, 0, 255),     # Buildings (blue)
           2 : (0, 255, 255),   # Low vegetation (cyan)
           3 : (0, 255, 0),     # Trees (green)
           4 : (255, 255, 0),   # Cars (yellow)
           5 : (255, 0, 0),     # Clutter (red)
        #    6 : (0, 0, 0)
           }       # Undefined (black)

def feature_vis(feats): # feaats形状: [b,c,h,w]
     output_shape = (256,256) # 输出形状
     channel_mean = torch.mean(feats,dim=1,keepdim=True) # channel_max,_ = torch.max(feats,dim=1,keepdim=True)
    #  channel_mean, _ = torch.max(feats,dim=1,keepdim=True)
     channel_mean = F.interpolate(channel_mean, size=output_shape, mode='bilinear', align_corners=False)
     channel_mean = channel_mean.squeeze(0).squeeze(0).cpu().numpy() # 四维压缩为二维
     channel_mean = (((channel_mean - np.min(channel_mean))/(np.max(channel_mean)-np.min(channel_mean)))*255).astype(np.uint8)
     savedir = '/home/lvhaitao/'
     if not os.path.exists(savedir+'feature_vis'): os.makedirs(savedir+'feature_vis') 
     channel_mean = cv2.applyColorMap(channel_mean, cv2.COLORMAP_JET)
     cv2.imwrite(savedir+'feature_vis/'+ '0.png',channel_mean)
     return channel_mean

def decode_segmentation(output, palette):
    output_class = torch.argmax(output, dim=1)  # 选择每个像素的最大类别
    decoded_images = torch.zeros((output_class.size(0), output_class.size(1), output_class.size(2), 3), dtype=torch.uint8)
    for class_idx, color in palette.items():
        mask = (output_class == class_idx)
        decoded_images[mask] = torch.tensor(color, dtype=torch.uint8)
    return decoded_images

index = 30
x1 = 877
y1 = 1244
dataset_dir = "/data/lvhaitao/dataset/Vaihingen/"

data = io.imread(dataset_dir+'top/top_mosaic_09cm_area{}.tif'.format(index))
data = 1 / 255 * np.asarray(data.transpose((2, 0, 1)), dtype='float32')
dsm = np.asarray(io.imread(dataset_dir+'dsm/dsm_09cm_matching_area{}.tif'.format(index)), dtype='float32')
min = np.min(dsm)
max = np.max(dsm)
dsm = (dsm - min) / (max - min)
x2 = x1 + 256
y2 = y1 + 256
data_p = data[:, x1:x2, y1:y2]
dsm_p = dsm[x1:x2, y1:y2]

net = convnextv2_unet_modify2.__dict__["convnextv2_unet_tiny"](
            num_classes=6,
            drop_path_rate=0.1,
            head_init_scale=0.001,
            use_orig_stem=False,
            heatmap=True,
            in_chans=3,
        ).cuda()
net.load_state_dict(torch.load("/home/lvhaitao/MyProject2/savemodel/MFFNet2_Vaihingen_epoch40_92.17164587537671"))
# net.load_state_dict(torch.load("/home/lvhaitao/MyProject2/savemodel/MFFNet3_Potsdam_epoch29_90.97468546733342"))



net.eval()

with torch.no_grad():
    data = torch.from_numpy(data_p).unsqueeze(0).cuda()
    dsm = torch.from_numpy(dsm_p).unsqueeze(0).cuda()
    output, heatmaps = net(data, dsm)
decoded_output = decode_segmentation(output, palette)
decoded_output = decoded_output.squeeze().cpu().numpy().astype(np.uint8)
heatmap1 = heatmaps[0].cpu()
heatmap2 = heatmaps[1].cpu()
heatmap3 = heatmaps[2].cpu()
heatmap4 = heatmaps[3].cpu()
heatmap_image = feature_vis(heatmap4)

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1), plt.title('label')
plt.imshow(decoded_output), plt.axis('off')
plt.subplot(1, 2, 2), plt.title('heatmap')
plt.imshow(heatmap_image), plt.axis('off')
plt.show()


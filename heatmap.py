import os
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from convnextv2 import convnextv2_unet_modify3
from skimage import io
from othermodel.SAGate import DeepLab, init_weight
import torch.autograd as autograd
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
from othermodel.CMGFNet import FuseNet

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

# index = 15
# x1 = 277
# y1 = 1133

index = 30
# x1 = 1530
# y1 = 544

x1 = 510
y1 = 1601
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

net = convnextv2_unet_modify3.__dict__["convnextv2_unet_tiny"](
            num_classes=6,
            drop_path_rate=0.1,
            use_orig_stem=False,
            heatmap=False,
            in_chans=3,
        ).cuda()
net.load_state_dict(torch.load("/home/lvhaitao/finetune/MFFNet(mixall+finetune_in_15_30)_Vaihingen_epoch1"))
# net.load_state_dict(torch.load("/home/lvhaitao/finetune/MFFNet(mixall)_vaihingen"))
# net.load_state_dict(torch.load("/home/lvhaitao/finetune/MFFNet(heatmap)_vaihingen_epoch20"))
# net.load_state_dict(torch.load("/home/lvhaitao/MyProject2/testsavemodel/MFFNet(mixall1)_Vaihingen_epoch47_92.25891413887855"))
# net.load_state_dict(torch.load("/home/lvhaitao/MyProject2/testsavemodel/MFFNet(mixall+noconvnextv2)_Potsdam_epoch46_91.86804260175617"))
# net.load_state_dict(torch.load("/home/lvhaitao/MyProject2/testsavemodel/MFFNet(mixall+nofef)_Potsdam_epoch32_92.09604204172777"))

#CMGFNet
# net = FuseNet(num_classes=6, pretrained=False).cuda()
# net.load_state_dict(torch.load("/home/lvhaitao/finetune/cmgfnet_vaihingen"))
# net.load_state_dict(torch.load("/home/lvhaitao/vaihingenmodel/CMGFNet_epoch22_91"))

net.eval()

data = torch.from_numpy(data_p).unsqueeze(0).cuda()
dsm = torch.from_numpy(dsm_p).unsqueeze(0).cuda()
input_tensor = torch.cat((data, dsm.unsqueeze(1)), dim=1)
output = net(input_tensor)
# print(output.shape)
decoded_output = decode_segmentation(output, palette)
decoded_output = decoded_output.squeeze().cpu().numpy().astype(np.uint8)
# Image.fromarray(decoded_output).save("decoded_output.png")

target_layers = [net.downsample_layers[-1]]
# target_layers = [net.sff_stage[-1]]
# target_layers = [net.enc_rgb5]

class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        
    def __call__(self, model_output):
        return (model_output[self.category, :, : ] * self.mask).sum()
    
normalized_masks = torch.nn.functional.softmax(output, dim=1).cpu()

sem_classes = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"]
sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
car_category = sem_class_to_idx["buildings"]

car_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
car_mask_uint8 = 255 * np.uint8(car_mask == car_category)
car_mask_float = np.float32(car_mask == car_category)
image = np.transpose(data_p, (1, 2, 0))
image = (image * 255).astype(np.uint8)
both_images = np.hstack((image, np.repeat(car_mask_uint8[:, :, None], 3, axis=-1)))
# both = Image.fromarray(both_images)
# both.save("both.png")
# both.show()

targets = [SemanticSegmentationTarget(car_category, car_mask_float)]
with GradCAM(model=net,target_layers=target_layers) as cam:
    # grayscale_cam = cam(input_tensor=input_tensor,targets=targets)
    grayscale_cam = cam(input_tensor=input_tensor,targets=targets)[0, :]
    rgb_img = np.float32(image) / 255
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

Image.fromarray(cam_image)    
# cam_image = Image.fromarray(cam_image)
# cam_image.save("cam_image.png")
# import matplotlib.pyplot as plt
# import numpy as np

# plt.figure(figsize=(6, 6))  # 设置图像大小
# plt.imshow(grayscale_cam, cmap='jet')  # 使用 jet 颜色映射
# plt.colorbar()  # 添加颜色条
# plt.axis('off')  # 隐藏坐标轴
# plt.title("Grad-CAM Heatmap")  # 添加标题
# plt.show()

# activation = {}
# def hook_fn(module, input, output):
#     activation['layer4_output'] = output
#     print(f"Layer4 output shape: {output.shape}")
# target_layers = target_layers[0]
# hook = target_layers.register_forward_hook(hook_fn)
# # 进行前向传播
# with torch.no_grad():
#     _ = net(data, dsm)
# # 取消 Hook
# hook.remove()

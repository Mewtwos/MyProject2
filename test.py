from PIL import Image
# from matplotlib import pyplot as plt
import numpy as np
import torch
from utils import convert_from_color
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
import torch.optim as optim


vaihingen_data = {}
vaihingen_dsm = {}
vaihingen_label = {}
patch_size = 256
# dataset = "Vaihingen"
dataset = "Potsdam"

if dataset == "Vaihingen":
    # index_dict = {"1":30, "2":30, "3":30, "4":30}
    # x_dict = {"1":2096, "2":1530, "3":510, "4":877}
    # y_dict = {"1":1441, "2":544, "3":1601, "4":1244}
    #test
    index_dict = {"1":15, "2":15}
    x_dict = {"1":822, "2":277}
    y_dict = {"1":1150, "2":1133}
    dataset_dir = "/data/lvhaitao/dataset/Vaihingen/"
else:
    # index_dict = {"1":"4_10", "2":"3_10", "3":"3_10", "4":"3_10"}
    # x_dict = {"1":3226, "2":853, "3":3406, "4":4747}
    # y_dict = {"1":3413, "2":5138, "3":2800, "4":2231}
    # index_dict = {"1":"3_10", "2":"3_10", "3":"3_10", "4":"3_10"}
    # x_dict = {"1":3828, "2":4716, "3":3406, "4":4747}
    # y_dict = {"1":4411, "2":5647, "3":2800, "4":2231}
    index_dict = {"1":"4_10", "2":"3_10", "3":"3_10", "4":"3_10"}
    x_dict = {"1":4934, "2":4716, "3":3406, "4":4747}
    y_dict = {"1":4816, "2":5647, "3":2800, "4":2231}
    dataset_dir = "/data/lvhaitao/dataset/Potsdam/"

for i in range(2):
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
    label = np.asarray(convert_from_color(io.imread('5_Labels_for_participants/top_potsdam_{}_label.tif')), dtype='int64')

    label_p = label[x1:x2, y1:y2]
    data_p = data[:, x1:x2, y1:y2]
    dsm_p = dsm[x1:x2, y1:y2]
    vaihingen_data[str(i+1)] = data_p
    vaihingen_dsm[str(i+1)] = dsm_p
    vaihingen_label[str(i+1)] = label_p


net = convnextv2_unet_modify3.__dict__["convnextv2_unet_tiny"](
            num_classes=6,
            drop_path_rate=0.1,
            patch_size=16,  ###原来是16
            use_orig_stem=False,
            in_chans=3,
        ).cuda()
# net.load_state_dict(torch.load("/home/lvhaitao/finetune/MFFNet(mixall)"))
# net.load_state_dict(torch.load("/home/lvhaitao/finetune/mffnet_potsdam"))
net.load_state_dict(torch.load("/home/lvhaitao/MyProject2/testsavemodel/MFFNet(mixlall+seed=20)_Potsdam_epoch32_90.47994674184432"))





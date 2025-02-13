import numpy as np
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from utils import *
from torch.autograd import Variable
from IPython.display import clear_output
from model.vitcross_seg_modeling import VisionTransformer as ViT_seg
from model.vitcross_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import wandb
# from othermodel.ukan import UKAN
# from othermodel.CMFNet import CMFNet
# from othermodel.rs3mamba import RS3Mamba, load_pretrained_ckpt
from othermodel.Transunet import VisionTransformer as TransUNet
# from othermodel.Transunet import CONFIGS as CONFIGS_ViT_seg
from convnextv2 import convnextv2_unet_modify, convnextv2_unet_modify2, convnextv2_unet_modify3, convnextv2_unet_modify4
from othermodel.MAResUNet import MAResUNet
from othermodel.ABCNet import ABCNet
from convnextv2.helpers import load_custom_checkpoint, load_imagenet_checkpoint
from othermodel.RFNet import RFNet, resnet18
from othermodel.ESANet import ESANet
from othermodel.ACNet import ACNet
from othermodel.SAGate import DeepLab, init_weight
from custom_repr import enable_custom_repr
from convnextv2.helpers import DiceLoss, SoftCrossEntropyLoss
from pynvml import *
enable_custom_repr()

use_wandb = True
if use_wandb:
    config = {
        "model": "MFFNet",
    }
    wandb.init(project="FTransUNet", config=config)
    wandb.run.name = "convnextv2-tiny-whuDataset-有权重-modify3(共享stage)-lla+spa+0.5diceloss+0.4auxloss"
    # wandb.run.name = "FTransUnet-Vaihingen-有权重2

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.cuda.device_count.cache_clear() 
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(int(os.environ["CUDA_VISIBLE_DEVICES"]))
print("Device :", nvmlDeviceGetName(handle))

seed = 3407
torch.manual_seed(seed)
random.seed(seed)  #新增
np.random.seed(seed)
torch.cuda.manual_seed_all(seed) #新增

#CMFNet
# net = CMFNet().cuda()
# vgg16_weights = torch.load('vgg16_bn-6c64b313.pth')
# mapped_weights = {}
# for k_vgg, k_segnet in zip(vgg16_weights.keys(), net.state_dict().keys()):
#     if "features" in k_vgg:
#         mapped_weights[k_segnet] = vgg16_weights[k_vgg]

# for it in net.state_dict().keys():
#     if it == 'conv1_1_d.weight':
#         avg = torch.mean(mapped_weights[it.replace('_d', '')].data, dim=1)
#         mapped_weights[it] = avg.unsqueeze(1)
#     if '_d' in it and it != 'conv1_1_d.weight':
#         if it.replace('_d', '') in mapped_weights:
#             mapped_weights[it] = mapped_weights[it.replace('_d', '')]
# try:
#     net.load_state_dict(mapped_weights)
#     print("Loaded VGG-16 weights in SegNet !")
# except:
#     pass

#rs3mamba
# net = RS3Mamba(num_classes=N_CLASSES).cuda()
# net = load_pretrained_ckpt(net)

#TransUNet
# config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
# config_vit.n_classes = 6
# config_vit.n_skip = 3
# config_vit.patches.grid = (14, 14)
# net = TransUNet(config_vit, 256, 6).cuda()
# net.load_from(weights=np.load(config_vit.pretrained_path))

#convnextv2_unet_modify
net = convnextv2_unet_modify3.__dict__["convnextv2_unet_tiny"](
            num_classes=8,
            drop_path_rate=0.1,
            patch_size=16,  
            use_orig_stem=False,
            in_chans=3,
        ).cuda()
# net = convnextv2_unet_modify4.__dict__["convnextv2_unet_tiny"](
#             num_classes=6,
#             drop_path_rate=0.1,
#             patch_size=16,  
#             use_orig_stem=False,
#             in_chans=3,
#         ).cuda()
net = load_imagenet_checkpoint(net, "/home/lvhaitao/pretrained_model/convnextv2_tiny_1k_224_fcmae.pt")
print("预训练权重加载完成")

#MAResUNet
#net = MAResUNet(num_classes=6).cuda()
#state_dict = net.state_dict()
#pretrained_dict = torch.load("/home/lvhaitao/.cache/torch/hub/checkpoints/resnet34-b627a593.pth")

#ABCNet
# net = ABCNet(6).cuda()

#Ftransunet
# config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
# config_vit.n_classes = 6
# config_vit.n_skip = 3
# config_vit.patches.grid = (int(256 / 16), int(256 / 16))
# net = ViT_seg(config_vit, img_size=256, num_classes=6).cuda()
# net.load_from(weights=np.load(config_vit.pretrained_path))

#RFNet
# resnet = resnet18(pretrained=True, efficient=False, use_bn=True)
# net = RFNet(resnet, num_classes=6, use_bn=True).cuda()

# ESANet
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

# ACNet
# net = ACNet(num_class=6, pretrained=True).cuda()

# SAGate
# pretrained_model = '/home/lvhaitao/resnet101_v1c.pth'
# net = DeepLab(6, pretrained_model=pretrained_model, norm_layer=nn.BatchNorm2d).cuda()
# init_weight(net.business_layer, nn.init.kaiming_normal_,nn.BatchNorm2d, 1e-5, 0.1,mode='fan_in', nonlinearity='relu')


params = 0
for name, param in net.named_parameters():
    params += param.nelement()
print(params)
if use_wandb:
    wandb.log({"params": params})

train_set = WHU_OPT_SARDataset(class_name='whu-opt-sar', root='/data/lvhaitao/dataset/whu-opt-sar/train')
train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
val_dataset = WHU_OPT_SARDataset(class_name='whu-opt-sar', root='/data/lvhaitao/dataset/whu-opt-sar/test')
val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

optimizer = optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.0005)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1)

def test(net, val_dataloader):
    with torch.no_grad():
        pred = []
        target = []
        net.eval()
        for idx, (sar, opt, label) in enumerate(tqdm(val_dataloader)):
            sar = sar.cuda()  
            opt = opt.cuda()
            label = label.cpu().numpy()
            outputs = net(opt, sar.squeeze(1))
            final_class = torch.argmax(outputs, dim=1)
            output = final_class.detach().cpu().numpy()
            for out, gt in zip(output, label):
                pred.append(out)
                target.append(gt)
        pred = np.concatenate([p.ravel() for p in pred])
        target = np.concatenate([t.ravel() for t in target])
        accuracy, mf1, miou, oa_dict = metrics(pred, target,
                                               label_values=['background', 'farmland', 'city', 'village', 'water','forest', 'road', 'others'])
        return accuracy, mf1, miou, oa_dict


def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch=1):
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    weights = weights.cuda()
    # diceloss = DiceLoss(smooth=0.05)
    # aux_loss = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=255)

    iter_ = 0
    acc_best = 89.0
    log_loss = 0
    for e in range(1, epochs + 1):
        net.train()
        for batch_idx, (sar, data, target) in enumerate(train_loader):
            data, sar, target = Variable(data.cuda()), Variable(sar.cuda()), Variable(target.cuda())
            optimizer.zero_grad()
            output, aux_out = net(data, sar.squeeze(1))
            loss = CrossEntropy2d(output, target)
            # loss = CrossEntropy2d(output, target, weight=weights) + 0.5 * diceloss(output, target) + 0.4 * aux_loss(aux_out, target)
            loss.backward()
            optimizer.step()

            losses[iter_] = loss.item()
            log_loss += loss.item()
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_])

            if iter_ % 500 == 0:
                clear_output()
                # rgb = np.asarray(255 * np.transpose(data.data.cpu().numpy()[0], (1, 2, 0)), dtype='uint8')
                pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                    e, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.item(), accuracy(pred, gt)))
            iter_ += 1

            del (data, target, loss)

        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
        if e > 0:
            net.eval()
            acc, mf1, miou, oa_dict = test(net, val_dataloader)
            net.train()
            if acc > acc_best:
                # torch.save(net.state_dict(), '/home/lvhaitao/MyProject2/testsavemodel/MFFNet(mixall1)_Vaihingen_epoch{}_{}'.format(e, acc))
                acc_best = acc
            if use_wandb:
                wandb.log({"epoch": e, "total_accuracy": acc, "train_loss": log_loss, "mF1": mf1, "mIoU": miou, "lr": current_lr, **oa_dict})
            log_loss = 0
    print('acc_best: ', acc_best)

#####   train   ####
time_start=time.time()
train(net, optimizer, 50, scheduler)
# test(net, val_dataloader)
time_end=time.time()
print('Total Time Cost: ',time_end-time_start)
if use_wandb:
    wandb.finish()


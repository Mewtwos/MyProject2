import numpy as np
from tqdm import tqdm
import time
import torch
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from convnextv2.helpers import load_custom_checkpoint
from utils import *
from torch.autograd import Variable
from IPython.display import clear_output
from convnextv2 import convnextv2_unet
import wandb

from custom_repr import enable_custom_repr
enable_custom_repr()

use_wandb = False
if use_wandb:
    config = {
        "model": "convnextv2_unet",
        "附加信息":"编码器独立，编码器特征融合"
    }
    wandb.init(project="FTransUNet", config=config)
    wandb.run.name = "convnextv2_unet-Vaihingen-atto-dwtaf_1layer-imagenet权重"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.device_count.cache_clear() 
os.environ["WORLD_SIZE"] = "1"
from pynvml import *
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(int(os.environ["CUDA_VISIBLE_DEVICES"]))
print("Device :", nvmlDeviceGetName(handle))

seed = 3407
torch.manual_seed(seed)
np.random.seed(seed)

net = convnextv2_unet.__dict__["convnextv2_unet_atto"](
            num_classes=6,
            drop_path_rate=0.1,
            head_init_scale=0.001,
            patch_size=16,  ###原来是16
            use_orig_stem=False,
            in_chans=3,
        )
print("开始加载权重")
net = load_custom_checkpoint(net, "/home/lvhaitao/convnextv2_atto_1k_224_fcmae.pt")
print("预训练权重加载完成")
# net.copy_all_parameters()
# print("编码器权重拷贝完成")
net.to(torch.device("cuda"))

params = 0
for name, param in net.named_parameters():
    params += param.nelement()
print(params)
if use_wandb:
    wandb.log({"params": params})

# Load the datasets
print("training : ", train_ids)
print("testing : ", test_ids)
print("BATCH_SIZE: ", BATCH_SIZE)
print("Stride Size: ", Stride_Size)
train_set = ISPRS_dataset(train_ids, cache=CACHE)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, drop_last=True, num_workers=8, pin_memory=True)

# base_lr = 0.01
# params_dict = dict(net.named_parameters())
# params = []
# for key, value in params_dict.items():
#     if '_D' in key:
#         # Decoder weights are trained at the nominal learning rate
#         params += [{'params': [value], 'lr': base_lr}]
#     else:
#         params += [{'params': [value], 'lr': base_lr}]

# optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
optimizer = optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.0005)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1)


def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    # Use the network on the test set
    ## Potsdam
    if DATASET == 'Potsdam':
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id))[:, :, :3], dtype='float32') for id in
                       test_ids)
        # test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id))[:, :, (3, 0, 1, 2)][:, :, :3],
        # dtype='float32') for id in test_ids)
    ## Vaihingen
    else:
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_dsms = (np.asarray(io.imread(DSM_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)
    eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)
    all_preds = []
    all_gts = []

    # Switch the network to inference mode
    with torch.no_grad():
        for img, dsm, gt, gt_e in tqdm(zip(test_images, test_dsms, test_labels, eroded_labels), total=len(test_ids),
                                       leave=False):
            pred = np.zeros(img.shape[:2] + (N_CLASSES,))

            total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
            for i, coords in enumerate(
                    tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total,
                         leave=False)):
                # Build the tensor
                image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                image_patches = np.asarray(image_patches)
                image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)

                min = np.min(dsm)
                max = np.max(dsm)
                dsm = (dsm - min) / (max - min)
                dsm_patches = [np.copy(dsm[x:x + w, y:y + h]) for x, y, w, h in coords]
                dsm_patches = np.asarray(dsm_patches)
                dsm_patches = Variable(torch.from_numpy(dsm_patches).cuda(), volatile=True)
                # Do the inference
                outs = net(image_patches, dsm_patches)
                outs = outs.data.cpu().numpy()

                # Fill in the results array
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out
                del (outs)

            pred = np.argmax(pred, axis=-1)
            all_preds.append(pred)
            all_gts.append(gt_e)
            clear_output()

    accuracy, mf1, miou, oa_dict = metrics(np.concatenate([p.ravel() for p in all_preds]),
                       np.concatenate([p.ravel() for p in all_gts]).ravel())
    if all:
        return accuracy, all_preds, all_gts
    else:
        return accuracy, mf1, miou, oa_dict


def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch=1):
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    weights = weights.cuda()

    iter_ = 0
    acc_best = 90.0

    for e in range(1, epochs + 1):
        # if e == 1:
        #     #解冻编码器
        #     print("Unfreezing the encoder part of the model")
        #     for param in net.parameters():
        #         param.requires_grad = True
        net.train()
        log_loss = 0
        for batch_idx, (data, dsm, target) in enumerate(train_loader):
            data, dsm, target = Variable(data.cuda()), Variable(dsm.cuda()), Variable(target.cuda())
            optimizer.zero_grad()
            output = net(data, dsm)
            loss = CrossEntropy2d(output, target, weight=weights)
            loss.backward()
            optimizer.step()

            losses[iter_] = loss.data
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_])

            if iter_ % 100 == 0:
                clear_output()
                rgb = np.asarray(255 * np.transpose(data.data.cpu().numpy()[0], (1, 2, 0)), dtype='uint8')
                pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                    e, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.data, accuracy(pred, gt)))
            iter_ += 1
            log_loss += loss.data
            del (data, target, loss)

            # if e % save_epoch == 0:
            # if iter_ % 500 == 0: #原来是500
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
        if e > 30:
            net.eval()
            acc, mf1, miou, oa_dict = test(net, test_ids, all=False, stride=Stride_Size)
            net.train()
            if acc > acc_best:
                # torch.save(net.state_dict(), '/home/lvhaitao/MyProject2/savemodel/_convnextv2_epoch{}_{}'.format(e, acc))
                acc_best = acc

            if use_wandb:
                wandb.log({"epoch": e, "total_accuracy": acc, "train_loss": log_loss, "mF1": mf1, "mIoU": miou, "lr": current_lr, **oa_dict})
            log_loss = 0
    print('acc_best: ', acc_best)


#####   train   ####
time_start = time.time()
train(net, optimizer, 50, scheduler)
time_end = time.time()
print('Total Time Cost: ', time_end - time_start)
if use_wandb:
    wandb.finish()


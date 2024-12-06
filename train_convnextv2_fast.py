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
from convnextv2 import convnextv2_mfnet
import wandb
import math
from new_dataset import Fast_ISPRS_dataset, ISPRS_Test_dataset

from custom_repr import enable_custom_repr
enable_custom_repr()

use_wandb = False 
if use_wandb:
    config = {
        "model": "convnextv2_mfnet_atto",
        "附加信息": "解码器固定256维度"
    }
    wandb.init(project="FTransUNet", config=config)
    wandb.run.name = "convnextv2_mfnet-Vaihingen-atto-编码器独立-解码器256-fast"


os.environ["CUDA_VISIBLE_DEVICES"] = "4"
torch.cuda.device_count.cache_clear() 
os.environ["WORLD_SIZE"] = "1"
from pynvml import *
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(int(os.environ["CUDA_VISIBLE_DEVICES"]))
print("Device :", nvmlDeviceGetName(handle))

seed = 3407
torch.manual_seed(seed)
np.random.seed(seed)

net = convnextv2_mfnet.__dict__["convnextv2_unet_atto"](
            num_classes=6,
            drop_path_rate=0.1,
            patch_size=16,
            use_orig_stem=False,
            in_chans=3,
        )
# net = load_custom_checkpoint(net, "/mnt/lpai-dione/ssai/cvg/workspace/nefu/lht/Main/results/MulEncoder/MulEncoder-FVit-checkpoint-50.pth")
# net = load_custom_checkpoint(net, "/mnt/lpai-dione/ssai/cvg/workspace/nefu/lht/FTransUNet/pretrainedmodel/mmearth1m-checkpoint-199.pth")
# print("预训练权重加载完成")
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
train_set = Fast_ISPRS_dataset(train_ids, cache=CACHE)
test_set = ISPRS_Test_dataset()
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, drop_last=True, num_workers=8, pin_memory=True)

base_lr = 0.01
# optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
optimizer = optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.0005)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1)


def adjust_learning_rate(epoch, optimizer, warmup_epochs=5, max_lr=1e-4, min_lr=1e-6, epochs=50):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        lr = max_lr * epoch / warmup_epochs
    else:
        lr = min_lr + (max_lr - min_lr) * 0.5 * (
            1.0
            + math.cos(
                math.pi
                * (epoch - warmup_epochs)
                / (epochs - warmup_epochs)
            )
        )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE, data_set=test_set):
    test_images = (x.clone for x in data_set.test_images)
    test_dsms = (x.clone for x in data_set.test_dsms)
    test_labels = data_set.test_labels
    eroded_labels = data_set.eroded_labels
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
                image_patches = [img[x:x + w, y:y + h].permute((2, 0, 1)) for x, y, w, h in coords]

                min = torch.min(dsm)
                max = torch.max(dsm)
                dsm = (dsm - min) / (max - min)
                dsm_patches = [dsm[x:x + w, y:y + h] for x, y, w, h in coords]
                # Do the inference
                outs = net(torch.stack(image_patches), torch.stack(dsm_patches))
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

    accuracy, mf1, miou = metrics(np.concatenate([p.ravel() for p in all_preds]),
                       np.concatenate([p.ravel() for p in all_gts]).ravel())
    if all:
        return accuracy, all_preds, all_gts
    else:
        return accuracy, mf1, miou


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
        net.eval()
        acc, mf1, miou = test(net, test_ids, all=False, stride=Stride_Size)
        net.train()
        if acc > acc_best:
            # torch.save(net.state_dict(), '/mnt/lpai-dione/ssai/cvg/workspace/nefu/lht/FTransUNet/savemodel/convnextv2_epoch{}_{}'.format(e, acc))
            acc_best = acc

        if scheduler is not None:
            scheduler.step()
            # adjust_learning_rate(e, optimizer)
            current_lr = optimizer.param_groups[0]['lr']

        if use_wandb:
            wandb.log({"epoch": e, "total_accuracy": acc, "train_loss": log_loss, "mF1": mf1, "mIoU": miou, "lr": current_lr})
        log_loss = 0
    print('acc_best: ', acc_best)


#####   train   ####
time_start = time.time()
# train(net, optimizer, 50, scheduler)
test(net, test_ids, all=False, stride=Stride_Size)
test(net, test_ids, all=False, stride=Stride_Size)
time_end = time.time()
print('Total Time Cost: ', time_end - time_start)
if use_wandb:
    wandb.finish()


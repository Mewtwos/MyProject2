from utils import *
import torch
from torchvision import transforms

data_files = [DATA_FOLDER.format(id) for id in train_ids]
dsm_files = [DSM_FOLDER.format(id) for id in train_ids]
label_files = [LABEL_FOLDER.format(id) for id in train_ids]
random_idx = 1

data = io.imread(data_files[1])
data = np.asarray(data.transpose((2, 0, 1)), dtype='float32')
dsm = np.asarray(io.imread(dsm_files[1]), dtype='float32')
min = np.min(dsm)
max = np.max(dsm)
dsm = (dsm - min) / (max - min) * 255
label = np.asarray((io.imread(label_files[random_idx])), dtype='int64')
x1, x2, y1, y2 = get_random_pos(data, WINDOW_SIZE)
data_p = data[:, x1:x2, y1:y2]
dsm_p = dsm[x1:x2, y1:y2]
label_p = label[x1:x2, y1:y2]
label_p = np.transpose(label_p, (2, 0, 1))

data_p = data_p.astype(np.uint8)
dsm_p = dsm_p.astype(np.uint8)
label_p = label_p.astype(np.uint8)
data_p = np.transpose(data_p, (1, 2, 0))
# dsm_p = np.transpose(dsm_p, (1, 2, 0))
label_p = np.transpose(label_p, (1, 2, 0))
Image_data = Image.fromarray(data_p)
Image_dsm = Image.fromarray(dsm_p)
Image_label = Image.fromarray(label_p)
#保存图片
Image_data.save('data_p.png')
Image_dsm.save('dsm_p.png')
Image_label.save('label_p.png')
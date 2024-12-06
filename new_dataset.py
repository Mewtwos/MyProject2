from utils import *

class Fast_ISPRS_dataset(torch.utils.data.Dataset):
    def __init__(self, ids, data_files=DATA_FOLDER, label_files=LABEL_FOLDER,
                 cache=False, augmentation=True):
        super(Fast_ISPRS_dataset, self).__init__()

        self.augmentation = augmentation
        self.cache = cache

        # List of files
        self.data_files = [DATA_FOLDER.format(id) for id in ids]
        # self.boundary_files = [BOUNDARY_FOLDER.format(id) for id in ids]
        self.dsm_files = [DSM_FOLDER.format(id) for id in ids]
        self.label_files = [LABEL_FOLDER.format(id) for id in ids]

        # Sanity check : raise an error if some files do not exist
        for f in self.data_files + self.dsm_files + self.label_files:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))

        # Initialize cache dicts
        self.data_cache_ = {}
        # self.boundary_cache_ = {}
        self.dsm_cache_ = {}
        self.label_cache_ = {}

        if cache:
            self.preload_data()
    
    def preload_data(self):
        for idx, data_file in enumerate(self.data_files):
            if DATASET == 'Potsdam':
                data = io.imread(data_file)[:, :, :3].transpose((2, 0, 1))
                data = 1 / 255 * np.asarray(data, dtype='float32')
            else:
                data = io.imread(data_file)
                data = 1 / 255 * np.asarray(data.transpose((2, 0, 1)), dtype='float32')
            self.data_cache_[idx] = data
            dsm = np.asarray(io.imread(self.dsm_files[idx]), dtype='float32')
            min = np.min(dsm)
            max = np.max(dsm)
            dsm = (dsm - min) / (max - min)
            self.dsm_cache_[idx] = dsm
            self.label_cache_[idx] = np.asarray(convert_from_color(io.imread(self.label_files[idx])), dtype='int64')

    def __len__(self):
        # Default epoch size is 10 000 samples
        return 10000
        if DATASET == 'Potsdam':
            return BATCH_SIZE * 1000
        elif DATASET == 'Vaihingen':
            return BATCH_SIZE * 1000
        else:
            return None

    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True

        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))

        return tuple(results)

    def __getitem__(self, i):
        # Pick a random image
        random_idx = random.randint(0, len(self.data_files) - 1)

        # Data is normalized in [0, 1]
        data = self.data_cache_[random_idx]
        dsm = self.dsm_cache_[random_idx]
        label = self.label_cache_[random_idx]

        # Get a random patch
        x1, x2, y1, y2 = get_random_pos(data, WINDOW_SIZE)
        data_p = data[:, x1:x2, y1:y2]
        dsm_p = dsm[x1:x2, y1:y2]
        label_p = label[x1:x2, y1:y2]

        # Data augmentation
        data_p, dsm_p, label_p = self.data_augmentation(data_p, dsm_p, label_p)

        # Return the torch.Tensor values
        return (torch.from_numpy(data_p),
                torch.from_numpy(dsm_p),
                torch.from_numpy(label_p))
    

class ISPRS_Test_dataset():
    def __init__(self):
        if DATASET == 'Potsdam':
            self.test_images = (torch.from_numpy(1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id))[:, :, :3], dtype='float32')).cuda() for id in
                        test_ids)
        ## Vaihingen
        else:
            self.test_images = (torch.from_numpy(1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32')).cuda() for id in test_ids)
        self.test_dsms = (torch.from_numpy(np.asarray(io.imread(DSM_FOLDER.format(id)), dtype='float32')).cuda() for id in test_ids)
        self.test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)
        self.eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)

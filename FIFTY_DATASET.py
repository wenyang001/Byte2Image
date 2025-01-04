import torch
from torch.utils.data import Dataset
import numpy as np
import os
import torchsnooper
from torchvision import transforms

ngram = 16

class FIFTY_DATASET(Dataset):
    def __init__(self, root, subset='train', block_size='512', scenario = '0', transform=None, target_transform=None, multi=True):

        super(FIFTY_DATASET, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.scenario = scenario
        self.block_size = block_size
        self.train = subset
        self.multi = multi
        self.data, self.targets, self.filename, self.labels = self.load(block_size, scenario, subset)

        # self.data = self.data.astype(np.uint)
        self.targets = self.targets.astype(np.int64)
        self.max_targets = self.targets.max() + 1

        print("Loaded {} data: data.shape={}, targets.shape={}".format(subset, self.data.shape, self.targets.shape))

    def data_add_bit(self, data):
        data_arr = []
        for shift_bit in range(8):
            tmp = self.getshift(shift_bit, data).astype(np.uint8)
            data_arr.append(tmp)

        data = np.array(data_arr)
        re = data
        for i in range(1, ngram):
            tmp = np.roll(data, -i)
            re = np.concatenate((re, tmp), axis=0)

        data = re
        data = data[:, :-ngram]
        data = np.expand_dims(data, 2)
        data = data.astype(np.uint8)
        return data

    def __getitem__(self, index):
        data, target, raw_data = self.data[index], self.targets[index], self.data[index]
        if self.multi:
            if self.block_size == '512':
                data = self.data_add_bit(data)
                t = transforms.Compose([
                    transforms.ToPILImage(mode='L'),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ])
                data = t(data)
                if self.transform is not None:
                    data = self.transform(data)

            else:
                L = [512*i for i in range(9)]
                t = transforms.Compose([
                    transforms.ToPILImage(mode='L'),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ])

                data_bit = []
                for i in range(8):
                    tmp = data[L[i]:L[i+1]]
                    tmp_bit = self.data_add_bit(tmp)
                    tmp_bit = t(tmp_bit)
                    data_bit.append(tmp_bit)

                data_bit = torch.cat(data_bit, axis=0)
                data = data_bit

                if self.transform is not None:
                    data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target, raw_data.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def getshift(self, shift, arr):
        arr_s = np.roll(arr, -1)
        arr_s[len(arr_s) - 1] = 0

        arr = arr << shift & 255
        arr_s = arr_s >> (8-shift) & 255

        arr = arr + arr_s
        return arr

    def getlabels(self):
        return self.labels

    def getfilename(self, index):
        return self.filename[index]

    def load(self, block_size='512', scenario='1', subset='train'):
        if block_size not in ['512', '4k']:
            raise ValueError('Invalid block size!')
        if subset not in ['train', 'val', 'test']:
            raise ValueError('Invalid subset!')

        data_dir = os.path.join(self.root, '{:s}_{:s}'.format(block_size, scenario))
        print(data_dir)
        if subset=='train':
            print(os.path.join(data_dir, '{}.npz'.format('train')))
            data = np.load(os.path.join(data_dir, '{}.npz'.format('train')))
        elif subset=='val':
            print(os.path.join(data_dir, '{}.npz'.format('val')))
            data = np.load(os.path.join(data_dir, '{}.npz'.format('val')))
        else:
            # test data
            print(os.path.join(data_dir, '{}.npz'.format('test')))
            data = np.load(os.path.join(data_dir, '{}.npz'.format('test')))

        data_x, data_y, data_z = data['x'], data['y'], []

        labels = {
          "1": ["jpg", "arw", "cr2", "dng", "gpr", "nef", "nrw", "orf", "pef", "raf", "rw2", "3fr", "tiff", "heic",
               "bmp", "gif", "png", "ai", "eps", "psd", "mov", "mp4", "3gp", "avi", "mkv", "ogv", "webm", "apk", "jar",
               "msi", "dmg", "7z", "bz2", "deb", "gz", "pkg", "rar", "rpm", "xz", "zip", "exe", "mach-o", "elf", "dll",
               "doc", "docx", "key", "ppt", "pptx", "xls", "xlsx", "djvu", "epub", "mobi", "pdf", "md", "rtf", "txt",
               "tex", "json", "html", "xml", "log", "csv", "aiff", "flac", "m4a", "mp3", "ogg", "wav", "wma", "pcap",
               "ttf", "dwg", "sqlite"],
         "2": ["bmp", "raw", "vec", "vid", "arc", "exe", "off", "pub", "hr", "aud", "oth"],
         "3": ["jpg", "arw", "cr2", "dng", "gpr", "nef", "nrw", "orf", "pef", "raf", "rw2", "3fr", "tiff", "heic",
               "bmp", "gif", "png", "mov", "mp4", "3gp", "avi", "mkv", "ogv", "webm", "oth"],
         "4": ["jpg", "raw", "vid", "5_bmps", "oth"],
         "5": ["jpg", "oth"],
         "6": ["jpg", "oth"],
         "tags": ["bitmap", "raw", "raw", "raw", "raw", "raw", "raw", "raw", "raw", "raw", "raw", "raw", "bitmap",
                  "bitmap", "bitmap", "bitmap", "bitmap", "vector", "vector", "vector", "video", "video", "video",
                  "video", "video", "video", "video", "archive", "archive", "archive", "archive", "archive", "archive",
                  "archive", "archive", "archive", "archive", "archive", "archive", "archive", "executable",
                  "executable", "executable", "executable", "office", "office", "office", "office", "office", "office",
                  "office", "published", "published", "published", "published", "human-readable", "human-readable",
                  "human-readable", "human-readable", "human-readable", "human-readable", "human-readable",
                  "human-readable", "human-readable", "audio", "audio", "audio", "audio", "audio", "audio", "audio",
                  "misc", "misc", "misc", "misc"]
         }

        return data_x, data_y, data_z, labels[scenario]

def arrayTobin(arr):
    for v in arr:
        print(f'{v:08b}', end=' ')



if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    train_transform = transforms.Compose([
        # transforms.ToTensor(),
        # transforms.ToPILImage(mode='L'),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomErasing(),
    ])

    train_dataset = FIFTY_DATASET('/media/liu/Data/DATASETS/', 'val', block_size='4k', scenario='6',
                                  multi=True,
                                  transform=train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=8,
        drop_last=True,
        shuffle=True)

    transform = transforms.ToPILImage(mode='L')
    for index, (x, y) in enumerate(train_loader):
        if index > 20:
            break
        x = x.squeeze(0)  # 压缩一维
        x = transform(x)  # 自动转换为0-255
        plt.imshow(x, cmap='gray')
        plt.show()



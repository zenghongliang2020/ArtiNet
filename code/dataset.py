import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class sapien_dateset(Dataset):

    def __init__(self, data_folder, train_type, img_size=256):
        self.data_folder = data_folder
        self.samples = os.listdir(data_folder)
        self.train_type = train_type
        self.img_size = img_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_folder = os.path.join(self.data_folder, self.samples[index])

        with Image.open(os.path.join(sample_folder, 'rgb.png')) as fimg:
            img = np.array(fimg.resize((self.img_size, self.img_size)), dtype=np.float32) / 255
        img = torch.from_numpy(img).permute(2, 0, 1)

        pcs_path = os.path.join(sample_folder, 'pcs.xyz')
        pcs = np.loadtxt(pcs_path, delimiter=' ', dtype=np.float32)
        idx = np.arange(pcs.shape[0])
        np.random.shuffle(idx)
        while len(idx) < 30000:
            idx = np.concatenate([idx, idx])
        idx = idx[:30000-1]
        pcs = pcs[idx, :]
        pcs = torch.from_numpy(pcs)

        if self.train_type == 'movable':
            label_path = os.path.join(sample_folder, 'labels.csv')
        elif self.train_type == 'joint_parameter':
            label_path = os.path.join(sample_folder, 'label_joint.txt')
        else:
            raise ValueError('train_type is unknown')
        label = pd.read_csv(label_path)
        label = torch.tensor(label['movable'])
        data_label = {'img': img, 'point_clouds': pcs, 'label': label}

        return data_label

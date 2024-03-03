from __future__ import division
import os
import random
import numpy as np
import cv2
from torch.utils import data
import sys
import matplotlib.pyplot as plt

sys.path.append("..")
from datasets.utils import *

class my_dataset(data.Dataset):
    def __init__(self, root, dataset='my', scene='K544', split='train',
                model='fdanet', aug='True'):
        self.intrinsics_color = np.array([[614.8531, 0.0, 319.6570],
                                        [0.0,     615.2770, 240.1615],
                                       [0.0,     0.0,  1.0]])
        self.intrinsics_color_inv = np.linalg.inv(self.intrinsics_color)            # 颜色相机内参的逆
        self.split = split
        self.data = ['no1', 'no2', 'no3', 'no4'] if self.split == 'train' else ['no4']
        self.aug = aug
        self.dataset = dataset
        self.frame = []
        self.RT = []
        for i in self.data:
            with open('/mnt/share/sda1/mnt/share/sda-8T/dk/Laser/K544/' + i + '/rgb.txt') as f:
                self.frames = f.readlines()
                for j in self.frames:
                    self.frame.append((i, j.split('_')[0]))

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        seq_id, id = self.frame[index]
        objs = {}
        objs['color'] = '/mnt/share/sda1/mnt/share/sda-8T/dk/Laser/K544/'+ seq_id + '/RGB' + '/' + f"{int(id):05d}" + '.png'
        objs['depth'] = '/mnt/share/sda1/mnt/share/sda-8T/dk/Laser/K544/'+ seq_id + '/Depth' + '/' + f"{int(id):05d}" + '.png'        # Twc
        objs['pose'] = '/mnt/share/sda1/mnt/share/sda-8T/dk/Laser/K544/'+ seq_id + '/Pose' + '/' + f"{int(id):05d}" + '.txt'
        try:
            img = cv2.imread(objs['color'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            print(objs['color'])
        pose = np.loadtxt(objs['pose'])
        
        if self.split == 'test':
            img, pose = to_tensor_query(img, pose)
            return img, pose

        pose[0:3,3] = pose[0:3,3] * 1000
        depth = cv2.imread(objs['depth'],-1)
        depth = cv2.resize(depth,(640,480))
        
        coord, mask = get_coord(depth, pose, self.intrinsics_color_inv)

        img, coord, mask = data_aug(img, coord, mask, self.aug)
        
        coord = coord[4::8,4::8,:]  # [60 80]
        mask = mask[4::8,4::8].astype(np.float16)

        img, coord, mask  = to_tensor(img, coord, mask)
        return img, coord, mask
if __name__ == '__main__':
    datat = my_dataset(split='train')
    trainloader = data.DataLoader(datat, batch_size=1, num_workers=1, shuffle=True, drop_last = True)

    for _, (img, coord, mask) in enumerate(trainloader):
        print(img.shape)
        print(coord.shape)
        print(mask.shape)


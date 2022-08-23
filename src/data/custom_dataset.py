# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 8/1/2022

import os
import cv2
import torch
import glob
import random
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import sampler
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, args):
        label_file = args['label_file']
        self.img_dir = args['img_dir']

        with open(label_file, 'r', encoding='utf8') as f:
            self.lines = f.readlines()
        self.imgH = args['imgH']
        self.imgW = 1200
        # self.list_image = glob.glob(os.path.join(root_dir, '*.[jp][pn]*'))
        transform_list = [#transforms.Grayscale(1),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5,), (0.5,))]
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return 3000  # len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        img_path, text = line.strip('\n').split('\t')
        # print(line)
        img_path = os.path.join(self.img_dir, img_path)
        if not os.path.exists(img_path):
            return self.__getitem__(random.randint(0, self.__len__() - 1))
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        ratio = w / float(h)
        imgW = int(np.floor(ratio * self.imgH))
        imgW = max(self.imgH, imgW)  # assure imgH >= imgW
        transform = resizeNormalize((imgW, self.imgH), self.imgW)
        img = transform(img)
        if self.transform is not None:
            img = self.transform(img)
        item = {'img': img, 'idx': idx}
        item['label'] = text.upper()
        return item


class CustomCollator(object):
    def __call__(self, batch):
        # img_path = [item['img_path'] for item in batch]
        width = [item['img'].shape[2] for item in batch]
        indexes = [item['idx'] for item in batch]

        imgs = torch.ones([len(batch), batch[0]['img'].shape[0], batch[0]['img'].shape[1], max(width)],
                          dtype=torch.float32)
        for idx, item in enumerate(batch):
            try:
                imgs[idx, :, :, 0:item['img'].shape[2]] = item['img']
            except:
                print(imgs.shape)
        item = {'img': imgs, 'idx': indexes}
        if 'label' in batch[0].keys():
            labels = [item['label'] for item in batch]
            item['label'] = labels
        return item


class resizeNormalize(object):
    def __init__(self, size, imgW, interpolation=Image.BILINEAR):
        self.size = size
        self.imgW = imgW
        self.interpolation = interpolation

    def __call__(self, img, padding=False):
        img = img.resize(self.size, self.interpolation)
        # if padding:
        #     img = self.padding_legnth(img, self.imgW)
        return img


if __name__ == '__main__':
    from config import args
    item = CustomDataset(args).__getitem__(4)
    print(item["img"].shape)
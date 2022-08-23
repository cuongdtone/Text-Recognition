# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 8/1/2022

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from src.models.crnn_mbv3 import CRNN
from src.data.custom_dataset import CustomDataset, resizeNormalize
from config import *
from src.utils.utils import OCRLabelConverter
from glob import glob


class TextRecognition:
    def __init__(self, args):
        self.net = CRNN(args)
        self.device = torch.device('cuda' if torch.cuda.is_available() and args['cuda'] else 'cpu')
        self.net.load_state_dict(torch.load(args['weight'], map_location=self.device)['state_dict'])
        self.net.to(self.device)
        self.net.eval()
        self.converter = OCRLabelConverter(alphabet)

        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5,), (0.5,))])
        self.imgH = args['imgH']
        self.imgW = 1200

    def __call__(self, img):
        """
        img is opencv img (bgr)
        """
        img = Image.fromarray(img[:, :, ::-1]).convert('RGB')
        w, h = img.size
        ratio = w / float(h)
        imgW = int(np.floor(ratio * self.imgH))
        imgW = max(self.imgH, imgW)  # assure imgH >= imgW
        resize = resizeNormalize((imgW, self.imgH), self.imgW)
        img = resize(img)
        img = self.transform(img)
        logits = self.net(img.unsqueeze(0))
        logits = logits.contiguous().cpu()
        logits = torch.nn.functional.softmax(logits, 2).argmax(2)
        pred = self.converter.decode(logits, torch.IntTensor([len(logits)]))
        return pred


if __name__ == '__main__':
    text_recognizer = TextRecognition(args)

    # path = r'D:\TextSpotter\vietnamese_dataset\img_crop'
    path = r'D:\TextSpotter\generator_scene_text\line_dataset\phase_1_step_0.05'
    list_images = glob(f'{path}/*.[jp][pn]*')
    for img_path in list_images:
        img = cv2.imread(img_path)
        pred = text_recognizer(img)
        print([pred])
        print(img_path, pred)
        cv2.imshow('cc', img)
        cv2.waitKey()


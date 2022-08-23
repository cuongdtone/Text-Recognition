# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 8/2/2022


import cv2
import numpy as np
import torch
from torchvision import transforms
import onnxruntime

from config import *
from src.utils.utils import OCRLabelConverter
from glob import glob


class TextRecognition:
    def __init__(self, weight='checkpoints/crnn_r18.onnx'):
        self.session = onnxruntime.InferenceSession(weight)
        self.device = torch.device('cuda' if torch.cuda.is_available() and args['cuda'] else 'cpu')

        self.converter = OCRLabelConverter(alphabet)

        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5,), (0.5,))])
        self.imgH = args['imgH']
        self.imgW = 1200
        self.input_mean = 0.5
        self.alphabet = alphabet

    def __call__(self, img):
        """
        img is opencv img (bgr)
        """
        h, w = img.shape[:2]

        ratio = w / float(h)
        imgW = int(np.floor(ratio * self.imgH))
        imgW = max(self.imgH, imgW)  # assure imgH >= imgW
        img = cv2.resize(img, (imgW, self.imgH))

        blob = cv2.dnn.blobFromImage(img, 1.0 / 127.5, (imgW, self.imgH),
                                      (self.input_mean, self.input_mean, self.input_mean), swapRB=True)

        logits = self.session.run(None, {'input': blob})[0]
        pred = self.decode(logits.T[0], [len(logits)])
        return pred

    def decode(self, t, length, raw=False):
        if len(length) == 1:
            length = length[0]
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            texts = []
            index = 0
            for i in range(len(length)):
                l = length[i]
                texts.append(
                    self.decode(t[index:index + l], [l], raw=raw))
                index += l
            return texts

if __name__ == '__main__':
    import random
    text_recognizer = TextRecognition()

    path = 'D:/TextSpotter/vietnamese_dataset/img_crop'
    list_images = glob(f'{path}/*.[jp][pn]*')
    random.shuffle(list_images)
    for img_path in list_images:
        img = cv2.imread(img_path)
        pred = text_recognizer(img)
        print(pred)
        cv2.imshow('cc', img)
        cv2.waitKey()


# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 7/29/2022


import sys
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.models import resnet18



class blockCNN(nn.Module):
    def __init__(self, in_nc, out_nc, kernel_size, padding, stride=1):
        super(blockCNN, self).__init__()
        self.in_nc = in_nc
        self.out_nc = out_nc
        self.kernel_size = kernel_size
        self.padding = padding
        # layers
        self.conv = nn.Conv2d(in_nc, out_nc,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm2d(out_nc)

    def forward(self, batch, use_bn=False, use_relu=False,
                use_maxpool=False, maxpool_kernelsize=None):
        """
            in:
                batch - [batch_size, in_nc, H, W]
            out:
                batch - [batch_size, out_nc, H', W']
        """
        batch = self.conv(batch)
        if use_bn:
            batch = self.bn(batch)
        if use_relu:
            batch = F.relu(batch)
        if use_maxpool:
            assert maxpool_kernelsize is not None
            batch = F.max_pool2d(batch, kernel_size=maxpool_kernelsize, stride=2)
        return batch



class blockRNN(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, bidirectional, dropout=0):
        super(blockRNN, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.bidirectional = bidirectional
        # layers
        self.gru = nn.LSTM(in_size, hidden_size, bidirectional=bidirectional)

    def forward(self, batch, add_output=False):
        """
        in array:
            batch - [seq_len , batch_size, in_size]
        out array:
            out - [seq_len , batch_size, out_size]
        """
        batch_size = batch.size(1)
        outputs, hidden = self.gru(batch)
        out_size = int(outputs.size(2) / 2)
        if add_output:
            outputs = outputs[:, :, :out_size] + outputs[:, :, out_size:]
        return outputs


class CRNN(nn.Module):
    def __init__(self, args,
                 bidirectional: bool = True,
                 dropout: float = 0.1):
        hidden_size = args['nHidden']
        vocab_size = args['nClasses']
        super(CRNN, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.bidirectional = bidirectional
        # make layers
        # convolutions
        resnet = resnet18(pretrained=True)
        modules = list(resnet.children())[:-3]

        self.resnet = nn.Sequential(*modules)

        self.cn6 = blockCNN(256, 256, kernel_size=3, padding=1)
        # RNN + Linear
        self.linear1 = nn.Linear(args['imgH'] * 16, 256)
        self.gru1 = blockRNN(256, hidden_size, hidden_size, bidirectional=bidirectional)
        self.gru2 = blockRNN(hidden_size, hidden_size, vocab_size, bidirectional=bidirectional)
        self.linear2 = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, batch: torch.Tensor, torch2onnx: bool = False):
        batch_size = batch.size(0)
        # convolutions
        batch = self.resnet(batch)
        # print(batch.shape)
        # batch = self.cn6(batch, use_relu=True, use_bn=True)
        # print(batch.shape)
        # make sequences of image features
        batch = batch.permute(0, 3, 1, 2)
        n_channels = batch.size(1)
        # print(batch_size, n_channels)
        batch = batch.view(batch_size, n_channels, -1)
        # print(batch.shape)
        batch = self.linear1(batch)
        # rnn layers
        batch = self.gru1(batch, add_output=True)
        batch = self.gru2(batch)
        # output
        batch = self.linear2(batch)
        batch = batch.permute(1, 0, 2)

        if torch2onnx:
            logits = batch.contiguous().cpu()
            logits = torch.nn.functional.softmax(logits, 2).argmax(2)
            return logits
        return batch


if __name__ == '__main__':
    from config import *
    net = CRNN(args)
    net.eval()
    import time
    x = torch.rand((32, 3, args['imgH'], 587))
    st = time.time()
    y = net(x)
    print(time.time() - st)
    print(y.shape)

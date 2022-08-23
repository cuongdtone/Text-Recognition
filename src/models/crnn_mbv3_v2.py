# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 8/23/2022


import torch.nn as nn
from torchvision.models import mobilenet_v3_large


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size], T = num_steps.
        output : contextual feature [batch_size x T x output_size]
        """
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output


class CRNN(nn.Module):

    def __init__(self, args, leaky_relu=False):
        super().__init__()
        img_h = args['imgH']
        num_channel = args['nChannels']
        num_class = args['nClasses']
        num_hidden = args['nHidden']

        mobilenetv3 = mobilenet_v3_large(pretrained=True)
        modules = list(mobilenetv3.children())[:-2]
        cnn = nn.Sequential(*modules)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        self.mobilenetv3 = nn.Sequential(*modules)

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(960, num_hidden, num_hidden),
            BidirectionalLSTM(num_hidden, num_hidden, num_class))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, 'the height of conv must be 1'
        conv = conv.squeeze(2)
        conv = conv.transpose(1, 2)  # [b, w, c]

        # rnn features
        output = self.rnn(conv)
        return output.permute(1, 0, 2)


if __name__ == '__main__':
    from config import *
    import torch
    import time
    args['imgH'] = 64
    net = CRNN(args)
    # print(net)
    net.eval()

    x = torch.rand((1, 3, 64, 587))
    st = time.time()
    y = net(x)
    print(time.time() - st)
    print(y.shape)


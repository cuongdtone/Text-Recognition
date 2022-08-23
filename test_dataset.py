# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 8/23/2022

import matplotlib.pyplot as plt
from src.data.custom_dataset import CustomDataset
from config import args

dataset = CustomDataset(args)
print(len(dataset))
data = dataset.__getitem__(1249)
print(data)
plt.imshow(data['img'].permute(1, 2, 0))
plt.show()

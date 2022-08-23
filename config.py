# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 8/1/2022

alphabet = """ !"$%&'()+,-./@0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_abcdefghijklmnopqrstuvwxyz{|}°²ÀÁÂÃÈÉÊÌÍÐÒÓÔÕÖÙÚÜÝàáâãèéêìíðòóôõöùúüýĀāĂăĐđĨĩŌōŨũŪūƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ–—’“”…€™−"""
alphabet = set(alphabet.upper())
alphabet = list(alphabet)
alphabet.sort()
alphabet = ''.join(alphabet)
print(alphabet)

args = {
    'name': 'exp1',
    'img_dir': r'D:\TextSpotter\generator_scene_text\line_dataset',
    'label_file': r'D:\TextSpotter\generator_scene_text\line_dataset\labels.txt',
    'imgH': 64,
    'nChannels': 3,
    'nHidden': 256,
    'nClasses': len(alphabet),
    'lr': 0.001,
    'epochs': 100,
    'batch_size': 32,
    'save_dir': 'checkpoints',
    'log_dir': 'logs',
    'resume': False,
    'cuda': False,
    'schedule': True,
    'weight': 'checkpoints/exp1/best.ckpt',
    'device': 'cpu'
}


import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import timm
from transform_model.poolformer import poolformer_s36


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class SimpleRes_512(nn.Module):
    def __init__(self, *, image_size, num_classes, in_channel=1):
        super().__init__()
        last_layer_num = 512
        self.CL1 = nn.Linear(512, 128)
        self.act = nn.ReLU(inplace=True)

        self.Former = poolformer_s36(num_classes=0, in_channel=in_channel, emb=64)
        self.CL2 = nn.Linear(128 + last_layer_num, num_classes)

    def forward(self, img_1d, img_2d):  #efficient 1280 res 512
        # 1d
        img_1d = self.CL1(img_1d)
        img_1d = self.act(img_1d)

        # 2d
        img_2d = self.Former(img_2d)

        # cat
        img_2d = torch.cat((img_1d, img_2d), dim=1)
        img_2d = self.CL2(img_2d)

        return img_2d  # 1 * 1000


class SimpleRes_4k(nn.Module):
    def __init__(self, *, image_size, num_classes, in_channel=8):
        super().__init__()
        last_layer_num = 512
        image_height, image_width = pair(image_size)
        self.CL1_ = nn.Linear(4096, 128)
        self.act_ = nn.ReLU(inplace=True)

        # self.Conv1 = nn.Conv2d(in_channel, 96, (image_height, 1), padding=(0, 0))
        self.Former = poolformer_s36(num_classes=0, in_channel=in_channel, emb=96)
        self.CL2_ = nn.Linear(128 + last_layer_num, num_classes)


    def forward(self, img_1d, img_2d):
        # 1d
        img_1d = self.CL1_(img_1d)
        img_1d = self.act_(img_1d)

        # 2d
        img_2d = self.Former(img_2d)

        # cat
        img_2d = torch.cat((img_1d, img_2d), dim=1)
        img_2d = self.CL2_(img_2d)

        return img_2d  # 1 * 1000


class SimpleRes_4k_1(nn.Module): # patch 32
    def __init__(self, *, image_size, num_classes, in_channel=8):
        super().__init__()
        last_layer_num = 512
        image_height, image_width = pair(image_size)
        self.CL1_ = nn.Linear(4096, 128)
        self.act_ = nn.ReLU(inplace=True)

        self.Former = poolformer_s36(num_classes=0, in_channel=in_channel, in_patch_size=32, in_stride=32)
        self.CL2_ = nn.Linear(128 + last_layer_num, num_classes)


    def forward(self, img_1d, img_2d):
        # 1d
        img_1d = self.CL1_(img_1d)
        img_1d = self.act_(img_1d)

        # 2d
        img_2d = self.Former(img_2d)

        # cat
        img_2d = torch.cat((img_1d, img_2d), dim=1)
        img_2d = self.CL2_(img_2d)

        return img_2d  # 1 * 1000

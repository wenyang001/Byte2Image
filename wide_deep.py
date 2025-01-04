
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import timm

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class SimpleRes_512(nn.Module):
    def __init__(self, *, image_size, num_classes, dim, channels=1, model_name='resnet18'):
        super().__init__()
        last_layer_num = 512
        image_height, image_width = pair(image_size)
        self.CL1 = nn.Linear(512, 128)
        self.act = nn.ReLU(inplace=True)
        self.Conv1 = nn.Conv2d(channels, dim, (image_height, 1), padding=(0, 0))
        self.Res = timm.create_model(model_name, num_classes=0, in_chans=1)
        self.CL2 = nn.Linear(128 + last_layer_num, num_classes)

    def forward(self, img_1d, img_2d):  #efficient 1280 res 512
        # 1d
        img_1d = self.CL1(img_1d)
        img_1d = self.act(img_1d)

        # 2d
        # print(img_2d.shape)
        img_2d = self.Conv1(img_2d).permute(0, 2, 1, 3)
        img_2d = self.Res(img_2d)
        # print(img_2d.shape)

        # cat
        img_2d = torch.cat((img_1d, img_2d), dim=1)
        img_2d = self.CL2(img_2d)

        return img_2d  # 1 * 1000


class SimpleRes_4k(nn.Module):
    def __init__(self, *, image_size, num_classes, dim, channels=1, model_name='resnet18'):
        super().__init__()
        last_layer_num = 512
        image_height, image_width = pair(image_size)
        self.CL1_ = nn.Linear(4096, 128)
        self.act_ = nn.ReLU(inplace=True)
        self.Conv1 = nn.Conv2d(channels, dim, (image_height, 1), padding=(0, 0))
        self.Res = timm.create_model(model_name, num_classes=0, in_chans=8)
        self.CL2_ = nn.Linear(128 + last_layer_num, num_classes)


    def forward(self, img_1d, img_2d):
        # 1d
        img_1d = self.CL1_(img_1d)
        img_1d = self.act_(img_1d)

        # 2d
        img_2d_per_c = torch.split(img_2d, 1, 1)
        input_x = []
        for i in range(8):
            tmp = self.Conv1(img_2d_per_c[i]).permute(0, 2, 1, 3)
            input_x.append(tmp)

        img_2d = torch.cat(input_x, dim=1)
        img_2d = self.Res(img_2d)

        # cat
        img_2d = torch.cat((img_1d, img_2d), dim=1)
        img_2d = self.CL2_(img_2d)

        return img_2d  # 1 * 1000


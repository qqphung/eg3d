#!/usr/bin/python
# -*- encoding: utf-8 -*-
from .model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2


class FaceParser:
    def __init__(self, device, ckpt_path="training/face_parser/segnet_ckpt.pth"):

        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.to(device)
        net.load_state_dict(torch.load(ckpt_path))
        net.eval()
        self.net = net
        
    def parse(self, image):
        '''
        image: B x 3 x H x W, torch.tensor
        '''
        with torch.no_grad():
            out = self.net(image)[0]
            parsing = out.argmax(1, keepdims=True)
        return parsing



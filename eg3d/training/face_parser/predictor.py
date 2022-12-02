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
    def __init__(self, ckpt_path="training/face_parser/segnet_ckpt.pth"):

        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.cuda()
        net.load_state_dict(torch.load(ckpt_path))
        net.eval()
        self.net = net

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    def parse(self, image):
        '''
        image: H x W x 3, np.array
        '''
        with torch.no_grad():
            img = Image.fromarray(image)
            image = img.resize((512, 512), Image.BILINEAR)
            img = self.to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = self.net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
        return parsing.copy().astype(np.uint8)



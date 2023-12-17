from __future__ import absolute_import

import os.path as osp
import random

from PIL import Image
from torchvision.transforms import InterpolationMode

# from DG_image.test_folder import DG_generate, img_generate
from reid.utils.data import transforms as T
import re
import os
from glob import glob

import numpy as np
from collections import defaultdict

dg_transforms = T.Compose([
            T.Resize((384, 128), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_style_img(pid,cam,sample_list):
    # pattern = re.compile(r'([-\d]+)_c([-\d]+)_')
    style_pid = pid
    style_cam = cam
    while style_pid == pid or style_cam == cam :
        style_img_root, style_pid, style_cam, _ = sample_list[random.randint(0,len(sample_list)-1)]
        # style_fname = os.path.basename(style_img_root)
        # style_pid, style_cam = map(int, pattern.search(style_fname).groups())
        # style_cam -= 1
        style_img = dg_transforms(Image.open(style_img_root).convert('RGB'))
    return style_img

class Preprocessor(object):
    def __init__(self, dataset, if_trans=False, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.sample_list = dataset
        self.if_trans = if_trans

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid, _ = self.dataset[index]
        fpath = fname

        if self.root is not None:
          fpath = osp.join(self.root, fname)
        
        # self.if_trans is True when we need generate the novel training samples in training stage 
        # self.if_trans is False in testing stage or in graph sampling stage 
        if self.if_trans:
            img_org = Image.open(fpath).convert('RGB')
            style_img_0 = get_style_img(pid, camid, self.sample_list)
            style_img_1 = get_style_img(pid, camid, self.sample_list)
            style_img_2 = get_style_img(pid, camid, self.sample_list)
            if self.transform is not None:
                img = self.transform(img_org)
                img_org = dg_transforms(img_org)
        else:
            img_org = Image.open(fpath).convert('RGB')
            if self.transform is not None:
                img = self.transform(img_org)
            style_img_0 = [0]
            style_img_1 = [0]
            style_img_2 = [0]
            img_org = [0]
            
        return img, img_org, style_img_0, style_img_1, style_img_2, fname, pid, camid

from __future__ import absolute_import, print_function

import os.path as osp
import re
from glob import glob

import numpy as np


class MSMT(object):

    def __init__(self, root, combine_all=True):

        self.root = root
        self.images_dir = root
        self.combine_all = combine_all
        self.train_path = 'bounding_box_train'
        self.gallery_path = 'bounding_box_test'
        self.query_path = 'query'
        self.train, self.val, self.query, self.gallery = [], [], [], []
        self.num_train_ids, self.num_val_ids, self.num_query_ids, self.num_gallery_ids = 0, 0, 0, 0
        self.has_time_info = False
        self.load()

    def preprocess(self, path, relabel=True):
        pattern = re.compile(r'([-\d]+)_c([-\d]+)_')
        all_pids = {}
        ret = []
        fpaths = sorted(glob(osp.join(self.root, path, '*.jpg')))
        for fpath in fpaths:
            fname = osp.basename(fpath)
            pid, cam = map(int, pattern.search(fname).groups())
            if pid == -1: continue
            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            pid = all_pids[pid]
            cam -= 1
            ret.append([fpath, pid, cam, 0])
        return ret, int(len(all_pids))

    def load(self):
        self.train, self.num_train_ids = self.preprocess(self.train_path)
        self.gallery, self.num_gallery_ids = self.preprocess(self.gallery_path, False)
        self.query, self.num_query_ids = self.preprocess(self.query_path, False)

        if self.combine_all:
            for item in self.train:
                item[0] = osp.join(self.train_path, item[0])
            for item in self.gallery:
                item[0] = osp.join(self.gallery_path, item[0])
                item[1] += self.num_train_ids
            for item in self.query:
                item[0] = osp.join(self.query_path, item[0])
                item[1] += self.num_train_ids
            self.train += self.gallery
            self.train += self.query
            self.num_train_ids += self.num_gallery_ids
            self.train_path = ''

        print(self.__class__.__name__, "dataset loaded")
        print("  subset   | # ids | # images")
        print("  ---------------------------")
        print("  train    | {:5d} | {:8d}"
              .format(self.num_train_ids, len(self.train)))
        print("  query    | {:5d} | {:8d}"
              .format(self.num_query_ids, len(self.query)))
        print("  gallery  | {:5d} | {:8d}"
              .format(self.num_gallery_ids, len(self.gallery)))

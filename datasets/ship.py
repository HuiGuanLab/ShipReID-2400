# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re

import os.path as osp

from .bases import BaseImageDataset
from collections import defaultdict
import pickle
class ShipReID2400(BaseImageDataset):
    """
    ShipReID2400

    Dataset statistics:
    # identities: 2400
    # images: 12988 (train) + 691 (val query) + 706 (test query) + 4253 (gallery)
    """
    dataset_dir = 'ShipReID-2400'

    def __init__(self, root='', dataset_dir=None, verbose=True, pid_begin = 0, **kwargs):
        super(ShipReID2400, self).__init__()
        if dataset_dir is not None:
            self.dataset_dir = dataset_dir
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.val_query_dir = osp.join(self.dataset_dir, 'val_query')
        self.test_query_dir = osp.join(self.dataset_dir, 'test_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()
        self.pid_begin = pid_begin
        train = self._process_dir(self.train_dir, relabel=True)
        test_query = self._process_dir(self.test_query_dir, relabel=False)
        val_query = self._process_dir(self.val_query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> ShipReID-2400 loaded")
            self.print_dataset_statistics(train, test_query, val_query, gallery)

        self.train = train
        self.test_query = test_query
        self.val_query = val_query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_test_query_pids, self.num_test_query_imgs, self.num_test_query_cams, self.num_test_query_vids = self.get_imagedata_info(self.test_query)
        self.num_val_query_pids, self.num_val_query_imgs, self.num_val_query_cams, self.num_val_query_vids = self.get_imagedata_info(self.val_query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.test_query_dir):
            raise RuntimeError("'{}' is not available".format(self.test_query_dir))
        if not osp.exists(self.val_query_dir):
            raise RuntimeError("'{}' is not available".format(self.val_query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self. gallery_dir))            

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d+)')

        pid_container = set()
        for img_path in sorted(img_paths):
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            assert 0 <= pid <= 2400  
            assert 1 <= camid <= 8
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]

            dataset.append((img_path, self.pid_begin + pid, camid, 0))
        return dataset
    
    def print_dataset_statistics(self, train, test_query, val_query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_test_query_pids, num_test_query_imgs, num_test_query_cams, num_train_views = self.get_imagedata_info(test_query)
        num_val_query_pids, num_val_query_imgs, num_val_query_cams, num_train_views = self.get_imagedata_info(val_query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  --------------------------------------------")
        print("  subset        | # ids | # images | # cameras")
        print("  --------------------------------------------")
        print("  train         | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  test_query    | {:5d} | {:8d} | {:9d}".format(num_test_query_pids, num_test_query_imgs, num_test_query_cams))
        print("  val_query     | {:5d} | {:8d} | {:9d}".format(num_val_query_pids, num_val_query_imgs, num_val_query_cams))
        print("  gallery       | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  --------------------------------------------")
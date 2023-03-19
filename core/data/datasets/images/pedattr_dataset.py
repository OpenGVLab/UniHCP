import os
import time
import pickle
import random
from easydict import EasyDict as edict
import numpy as np
import torch.utils.data as data
from PIL import Image
from core.data.transforms.pedattr_transforms import PedAttrAugmentation, PedAttrTestAugmentation, PedAttrRandomAugmentation
from core import distributed_utils as dist


__all__ = ['AttrDataset']

class AttrDataset(data.Dataset):

    def __init__(self, ginfo, augmentation, task_spec, train=True, **kwargs):

        assert task_spec.dataset in ['peta', 'PA-100k', 'rap', 'rap2', 'uavhuman', 'HARDHC', 'ClothingAttribute', 'parse27k', 'duke', 'market'], \
            f'dataset name {task_spec.dataset} is not exist'

        data_path = task_spec.data_path

        with open(data_path, "rb+") as f:
            dataset_info = pickle.load(f)
        dataset_info = edict(dataset_info)

        img_id = dataset_info.image_name
        attr_label = dataset_info.label

        if train:
            split = 'trainval'
        else:
            split = 'test'

        assert split in dataset_info.partition.keys(), f'split {split} is not exist'

        height = augmentation.height
        width = augmentation.width

        self.dataset = task_spec.dataset
        self.root_path = task_spec.root_path

        if train:
            self.transform = PedAttrAugmentation(height, width)
            if augmentation.get('use_random_aug', False):
                self.transform = PedAttrRandomAugmentation(height, width, \
                    augmentation.use_random_aug.m, augmentation.use_random_aug.n)
        else:
            self.transform = PedAttrTestAugmentation(height, width)

        self.attr_id = dataset_info.attr_name
        self.attr_num = len(self.attr_id)

        self.img_idx = dataset_info.partition[split]

        if isinstance(self.img_idx, list):
            self.img_idx = self.img_idx[0]  # default partition 0

        self.img_num = len(self.img_idx)
        self.img_idx = np.array(self.img_idx)
        self.img_id = [img_id[i] for i in self.img_idx]
        self.label = attr_label[self.img_idx]
        self.task_name = ginfo.task_name
        self.rank = dist.get_rank()
        self.train = train

    def __getitem__(self, index):
        imgname, gt_label, imgidx = self.img_id[index], self.label[index], self.img_idx[index]
        imgpath = os.path.join(self.root_path, imgname)

        img = Image.open(imgpath).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        gt_label = gt_label.astype(np.float32)

        output = {'image': img, 'label': gt_label, 'filename': imgname}
        return output

    def __len__(self):
        return len(self.img_id)

    def __repr__(self):
        return self.__class__.__name__ + \
               f'rank: {self.rank} task: {self.task_name} mode:{"training" if self.train else "inference"} ' \
               f'dataset_len:{len(self.img_id)} id_num:{self.attr_num} augmentation: {self.transform}'


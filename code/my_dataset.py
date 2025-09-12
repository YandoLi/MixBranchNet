import torch
from monai import transforms
from monai.transforms import apply_transform
from torch.utils.data import Dataset
import scipy.io as io
import numpy as np
from utils import read_data


class MyDataSet(Dataset):
    """自定义数据集
        输入：
            images_path：数据的路径
            images_class：
            transform：对数据做的变换，一般用作数据增强和格式转换
            channels：数据的维度，eg：对 RGB维度为 3，对 4pools为 4
            use_cache:是否使用缓存，具体原理不清楚
    """

    def __init__(self, images_path: list,
                 img_transform=None, label_transform=None,
                 channels=(0, 1, 2, 3, 4, 5, 6, 7, 8),
                 use_cache=False):
        self.images_path = images_path
        # self.images_class = images_class
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.channels = channels
        self.use_cache = use_cache
        self.cache_data = []
        self.cache_label = []

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        if not self.use_cache:
            temp = io.loadmat(self.images_path[item])['savedata']
            label = io.loadmat(self.images_path[item])['savedata_mask']
            # label = self.images_class[item]
            img = temp[self.channels, ...]
            if img.shape[1] == 1:
                img = np.transpose(img, (1, 0))

            if self.img_transform is not None:
                # img = self.transform(img)
                img = apply_transform(self.img_transform, img.astype(np.float64)).float()
            if self.label_transform is not None:
                label = self.label_transform(label, dtype=torch.int64)

            self.cache_data.append(img)
            self.cache_label.append(label)
        else:
            img = self.cache_data[item]
            label = self.cache_label[item]
        return img, label

    def set_use_cache(self, use_cache):
        """set_use_cache
            在实例化的 MyDataSet实例中的 use_cache参量设置为 True时起作用
            具体作用：
                如果use_cache设置为True，就将 self.cache_data 和 self.cache_label转换为tuple(元组)数据类型
                如果use_cache设置为了False,就将 self.cache_data 和 self.cache_label设置为空
        """
        if use_cache:
            x_img = tuple(self.cache_data)  # 转换为元组tuple数据类型

            if x_img:
                self.cache_data = torch.stack(x_img)  # 堆叠
                lab_img = tuple(self.cache_label)
                self.cache_label = tuple(lab_img)
            else:
                print(x_img)

        else:
            self.cache_data = []
            self.cache_label = []
        self.use_cache = use_cache

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets
        # images, labels = tuple(zip(*batch))
        # images = torch.stack(images, dim=0)
        # labels = torch.as_tensor(labels)
        # return images, labels


def cat_list(images, fill_value=0):
    # 计算该batch数据中，channel, h, w的最大值
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


def dataset_test(path, r_seed, channel):
    images_path = read_data(root=path, random_seed=r_seed)

    test_transform = transforms.ToTensor()
    # 实例化测试数据集
    dataset = MyDataSet(images_path=images_path,
                        img_transform=test_transform,
                        label_transform=torch.as_tensor,
                        channels=channel)
    return dataset, images_path

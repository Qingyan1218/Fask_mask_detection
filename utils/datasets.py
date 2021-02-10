import glob
import random
import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils.data_aug import mosaic, cut_mix


def pad_to_square(img, pad_value):
    # 将图像填充至正方形
    c, h, w = img.shape
    # 计算差值
    dim_diff = np.abs(h - w)
    # 左、上的填充值，右、下的填充值
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # 如果h<w，则填充h，否则填充w，pad采用（左，右，上，下）
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # 填充
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    """采用最近领插值进行图像缩放"""
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    """随机缩放，从288到448中每隔32，随机取一个值"""
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        # 寻找目标文件夹下的所有文件
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # 将图像转变成tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # 进行0填充
        img, _ = pad_to_square(img, 0)
        # resize到指定大小
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True,
                 multiscale=True, normalized_labels=True, mode='baseline'):
        with open(list_path, "r") as file:
            # 这里的list_path里应该全是图片的路径
            self.img_files = file.readlines()
        # label_files的路径和img_files一致，因此将图片变成labels，将后缀变成txt
        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]

        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.mode = mode

    def __getitem__(self, index):
        # index % len(self.img_files)仍为index
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        label_path = self.label_files[index % len(self.img_files)].rstrip()
        if self.mode == 'baseline':
            # 打开图像准变成tensor
            img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
            # 处理少于三个通道的图片
            if len(img.shape) != 3:
                img = img.unsqueeze(0)
                img = img.expand((3, img.shape[1:]))
            c, h, w = img.shape
            # label的路径，获得归一化后的标签
            if os.path.exists(label_path):
                boxes = np.loadtxt(label_path).reshape(-1, 5)

        elif self.mode == 'mosaic':
            img, boxes = mosaic(self.img_size, self.img_files, self.label_files, index)
            img = transforms.ToTensor()(img)
            c, h, w = img.shape

        elif self.mode == 'cutmix':
            img, boxes = cut_mix(self.img_files, self.label_files, index)
            img = transforms.ToTensor()(img)
            c, h, w = img.shape
        else:
            raise ValueError("no mode named %s, mode must be in ['baselie','mosaic','cutmix']" % self.mode)

        # 如果label进行过归一化，则缩放因子是图像尺寸本身，否则就是不缩放，本模型均采用缩放
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # 短边填充，变成方形
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # 抽取未pad未scale的坐标，乘以缩放因子即可恢复到原图上
        x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
        y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
        x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
        y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
        # 加上pad
        x1 += pad[0]
        y1 += pad[2]
        x2 += pad[1]
        y2 += pad[3]
        # 返回归一化后的坐标，padded_w和padded_h即图像长边的尺寸
        # 中心点归一化到padded_w和padded_h尺度上
        boxes[:, 1] = ((x1 + x2) / 2) / padded_w
        boxes[:, 2] = ((y1 + y2) / 2) / padded_h
        # 先乘以w_factor和h_factor放大到原图尺寸，再归一化到padded_w和padded_h尺度上
        boxes[:, 3] *= w_factor / padded_w
        boxes[:, 4] *= h_factor / padded_h

        targets = torch.zeros((len(boxes), 6))
        # targets的第0位是0，在collate_fn中使用
        targets[:, 1:] = torch.from_numpy(boxes)

        # 数据增广，水平翻转
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)
        return img_path, img, targets

    def collate_fn(self, batch):
        """制定规则，提供给DataLoader抽取样本"""
        paths, imgs, targets = list(zip(*batch))
        # 移除空的targets
        targets = [boxes for boxes in targets if boxes is not None]
        # 将targets的第一位变成序号，即代表batch中的第几个样本，因为不同的样本所拥有的box数量不同，
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # 每10个batch选取新的图像尺寸
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # 因为bbox尺寸已经归一化，所以不用缩放
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)




from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size, device = None):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    # Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    Tensor = torch.cuda.FloatTensor if device==torch.device('cuda') else torch.FloatTensor
    labels = []
    # 列表元素为(TP, confs, pred)
    sample_metrics = []
    for batch_i, (_, imgs, targets) in enumerate(dataloader):

        # 将所有的标签拿出来
        labels += targets[:, 1].tolist()
        # 转换坐标并放大到原始图像尺寸
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        # 获取每个批次的统计数据
        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # 统计数据组合到一起，list(zip(*sample_metrics))将TP，conf，pred分别组成一个元组
    # true_positives.shape, pred_scores.shape, pred_labels.shape均为(n,)，n为所有图片框的总数

    ## 因为sample_metrics可能是空，所以这里进行判断，源代码只有59和60行
    if len(sample_metrics):
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    else:
        ## 随便给出一个全零矩阵，可以计算np.mean即可
        print('sample_metrics is []')
        precision=np.zeros((2,1))
        recall=np.zeros((2,1))
        AP=np.zeros((2,1))
        f1=np.zeros((2,1))
        ap_class=[]
    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3_tiny_face_mask.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/face_mask.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_50_cutmix.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/face_mask.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # 初始化模型
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # 加载darknet weights
        print('loading darknet weights')
        model.load_darknet_weights(opt.weights_path)
    else:
        # 加载checkpoint weights
        print('loading checkpoint weights')
        model.load_state_dict(torch.load(opt.weights_path, map_location=torch.device('cpu')))

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=8,
        device = device
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")

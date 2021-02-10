from models import *
# from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

from terminaltables import AsciiTable

import os
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3_tiny_face_mask.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/face_mask.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=10, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--augmentation_mode", default='baseline', help="allow different augmentation for dataset")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    opt = parser.parse_args()
    # print(opt)
    print(opt.augmentation_mode)

    # logger = Logger("logs")
    # 采用yolov3_simple,batchsize可以到2，直接采用yolov3的话，batchsize为1都无法用GPU
    # 采用yolov3_tiny_face_mask，batchsize可以到4
    # mac上batchsize=2，100个batch需要18分钟
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # parse_data_config解析文件
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # 初始化模型
    model = Darknet(opt.model_def, loss_mode='baseline', reg_mode='smooth').to(device)
    model.apply(weights_init_normal)

    # 加载保存的模型参数
    # opt.pretrained_weights = './checkpoints/yolov3_ckpt_50_cutmix.pth'
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            print('load pretrained weights......')
            # model.load_state_dict(torch.load(opt.pretrained_weights))
            # 在仅有cpu的机器上，加上map_location
            model.load_state_dict(torch.load(opt.pretrained_weights, map_location=torch.device('cpu')))
            print('load pretrained weights successfully')
        else:
            print('load darknet weights......')
            model.load_darknet_weights(opt.pretrained_weights)
            print('load darknet weights successfully')

    # 加载数据，默认normalized_labels=True，即原始labels是归一化后的
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training, mode=opt.augmentation_mode)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    print('training begin......')
    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()
        for batch_i, (img_path, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i
            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)
            # 采用cutmix或者mosaic进行训练
            # outputs.shape：torch.Size([batchsize, w*h, 7])
            loss, outputs = model(imgs, targets)
            loss.backward()

            # 积累一定的loss之后进行梯度反传
            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch+1, opt.epochs, batch_i+1, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # 记录模型指标
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # 利用Tensorboard展示结果
                # tensorboard --logdir='logs' --port=6006，网页中打开localhost:6006
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                # print(tensorboard_log)
                # logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # 计算剩余时间
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            if (batch_i+1) % 50 == 0:
                # print("---- [Epoch %d/%d, Batch %d/%d] ----" % (epoch+1, opt.epochs, batch_i+1, len(dataloader)))
                # print(loss)
                print(log_str)

            model.seen += imgs.size(0)

        if (epoch+1) % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # 验证
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=4,
                device=device
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            print(evaluation_metrics)
            # logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]

            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

        if (epoch+1) % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d_%s.pth" % (epoch+1, opt.augmentation_mode))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.parse_config import *
from utils.utils import build_iou_targets, to_cpu
from utils.activation_and_loss import Swish, Mish, iou_loss, LabelSmoothingCrossEntropy, DropBlock2D


"""本次修改在原来models.py的基础增加激活函数的选择及正则化
原来的models.py可参见CV->22->week22homework
激活函数的选择在yolov3.cfg中修改activation=leaky，
loss的选择在Darknet中增加loss_mode
regularizaion选择在Darknet中增加reg_mode"""
from utils.activation_and_loss import Swish, Mish

def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    # 第0层是超参，因此弹出赋值给hyperparams
    hyperparams = module_defs.pop(0)
    # output_filters 初始值为[3]，后续增加上一层的输出channel
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()
        # 增加卷积层
        if module_def["type"] == "convolutional":
            # bn的指示，等于1表示需要bn
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            if module_def["activation"] == 'swish':
                modules.add_module(f"swish_{module_i}", Swish())
            elif module_def["activation"] == 'mish':
                modules.add_module(f"mish_{module_i}", Mish())
            else:
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))

        # 增加池化层
        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)

        # 增加上采样层
        elif module_def["type"] == "upsample":
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)

        # 增加route层
        elif module_def["type"] == "route":
            # route层是融合层，输出channel是不同层的相加
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(f"route_{module_i}", EmptyLayer())

        # 增加shortcut层
        elif module_def["type"] == "shortcut":
            # 选出shortcut开始的那一层的输出channel
            # 一般是漏斗型，因此每隔3层进行一次shortcut
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", EmptyLayer())

        # 增加yolo层，共3次
        elif module_def["type"] == "yolo":
            # [6,7,8]->[3,4,5]->[0,1,2]
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # 抽取anchors，三次均为[10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326]
            # anchors代表框的宽和高
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            # 每两个抽出来组成元组，共9组
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            # 先抽出[6,7,8]处的3组，[(116, 90), (156, 198), (373, 326)],依次类推
            anchors = [anchors[i] for i in anchor_idxs]
            # 共有80类
            num_classes = int(module_def["classes"])
            # 超参，416
            img_size = int(hyperparams["height"])
            # 定义检测层
            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
            modules.add_module(f"yolo_{module_i}", yolo_layer)

        module_list.append(modules)
        # 增加本层的输出channel
        output_filters.append(filters)

    return hyperparams, module_list


class Upsample(nn.Module):
    """ nn.Upsample被移除了，
     采用最近邻插值进行上采样"""

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class EmptyLayer(nn.Module):
    """'route' 和 'shortcut' 层的占位符
    直接去取之前计算的结果，不需要进行计算"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):
    """检测层"""

    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        # 刚开始为[(116, 90), (156, 198), (373, 326)]
        self.anchors = anchors
        # 3
        self.num_anchors = len(anchors)
        # 80
        self.num_classes = num_classes
        # 阈值，大于该阈值的iou处认为是有物体的，仅在noobj_mask中设置
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        # 损失计算的比例
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0

    def compute_grid_offsets(self, grid_size, cuda=True):
        # grid_size分别是h/32, h/16, h/8
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        # 计算当前的缩放比例
        self.stride = self.img_dim / self.grid_size
        # self.grid_x是[1, 1, g, g]的矩阵，最后一个g中是0~g-1，相当于打了网格
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        # 将3个anchor分别缩放到和特征图尺度一致
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        # w和h方向anchor的尺寸,shape=torch.Size([batchsize, 3, 1, 1])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None, loss_mode = 'ciou', reg_mode = 'bce'):

        # 定义一些Tensor类型
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor
        self.img_dim = img_dim
        # num_samples是batchsize,x的shape
        # torch.Size([batchsize, 255, h/32, w/32])
        # torch.Size([batchsize, 255, h/16, w/16])
        # torch.Size([batchsize, 255, h/8, w/8])
        num_samples = x.size(0)
        # x的h，每个图像都不一致
        grid_size = x.size(2)

        # 将x变形,3代表3个anchor，每个anchor用85维去预测box，confidence和80分类
        # torch.Size([batchsize, 3, h/32, w/32, 85])
        # torch.Size([batchsize, 3, h/16, w/16, 85])
        # torch.Size([batchsize, 3, h/8, w/8, 85])
        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )


        # 输出，Center x，Center y，Width，Height，pred_conf，pred_cls
        # x,y,w,h的维度torch.Size([batchsize, 3, h/32, w/32])->h/16, w.16->h/8, w/8
        # x和y经过sigmoid变成0~1，即表示中心点与网格点的偏移
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        # pred_conf和pred_cls经过sigmoid表示输出一个概率值
        # pred_conf代表是不是物体的概率乘以IOU
        pred_conf = torch.sigmoid(prediction[..., 4])
        # torch.Size([batchsize, 3, h/32, w/32， 80])->h/16, w.16->h/8, w/8
        # yolov3为了预测多标签，将softmax修改为sigmoid，改成softmax后不影响模型的加载
        # sigmoid可以针对每个值进行计算，因此无需指定维度，但是softmax必须指定维度
        pred_cls = torch.sigmoid(prediction[..., 5:])
        # pred_cls = torch.softmax(prediction[..., 5:],dim=4)

        # 如果grid size和当前的不匹配，重新计算偏移，初始的时候是0，因此不匹配
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # 增加anchors的偏移和缩放，即每个尺度下每个像素点预测3个框，
        # pred_boxes.shape = torch.Size([batchsize, 3, h/32, w/32, 4])->h/16, w.16->h/8, w/8
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        # 第0和1位是偏移值，加上所在的格子坐标就是box的坐标
        # x,y的维度和self.grid_x，self.grid_y仅channel不一致
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        # 第2和3位是w和h，从预测值转换成box的实际值
        # w，h的维度和self.anchor_w，self.anchor_h的维度后两位不一致
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        # 将结果组合到一起，self.stride由self.compute_grid_offsets()得到，是当前的缩放倍数
        # output.shape=torch.Size([batchsize, 3 x h/32 x w/32, 85])->h/16, w/16->h/8, w/8
        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        if targets is None:
            return output, 0
        else:
            # 将targets转变,targets是n行6列的矩阵，targets[:, 1:] = boxes，n是groundtruth boxes的个数
            # 对于obj_mask,1一定是正样本落在的地方，0一定不是正样本落在的地方
            # 对于noobj_mask，1一定是负样本落在的地方，0不一定是正样本落在的地方，也可能是不参与计算
            # 增加CIOU的计算，因此采用build_ciou_targets
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf, t_box = build_iou_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )
            # 除了conf loss 需要计算noobj，其余均不需要，obj_mask即有物体中心的地方
            obj_mask = obj_mask.bool()
            noobj_mask = noobj_mask.bool()
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])

            if loss_mode == 'baseline':
                # loss_loc代表位置的损失
                loss_loc = loss_x + loss_y + loss_w + loss_h
            elif loss_mode in ['iou', 'ciou', 'diou', 'giou']:
                # 计算各种IOU_LOSS, pred_boxes已经是还原到原图上的坐标，b_box是gt还原到原图的坐标
                # pred_boxes:torch.Size([batchsize, 3 , h/32 , w/32, 4])->h/16, w/16->h/8, w/8
                loss_loc = torch.mean(iou_loss(pred_boxes[obj_mask], t_box[obj_mask], iou_mode=loss_mode))
            else:
                raise ValueError('no loss_mode named %s' % loss_mode)

            # 对于noobj，基本都能预测对，因此loss_conf_noobj通常比较小
            # 为了平衡，noobj_scale通常大于obj_scale
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            if reg_mode == 'smooth' or reg_mode == 'all':
                # 取出最后一维中最大值所在的位置
                tcls = torch.argmax(tcls[obj_mask], dim=-1).long()
                loss_cls = LabelSmoothingCrossEntropy()(pred_cls[obj_mask], tcls, smoothing=0.1)
            elif reg_mode == 'baceline':
                loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            else:
                raise ValueError('no regularization mode named %s' % reg_mode)
            total_loss = loss_loc + loss_conf +loss_cls

            # 分类正确的百分比，class_mask[obj_mask]即分类正确的地方为1
            cls_acc = 100 * class_mask[obj_mask].mean()
            # 有物体中心的地方预测到的confidence的均值
            conf_obj = pred_conf[obj_mask].mean()
            # 无物体中心的地方预测到的confidence的均值
            conf_noobj = pred_conf[noobj_mask].mean()
            # 预测confidence大于0.5的地方为1
            conf50 = (pred_conf > 0.5).float()
            # iou大于0.5的地方为1
            iou50 = (iou_scores > 0.5).float()
            # iou大于0.75的地方为1
            iou75 = (iou_scores > 0.75).float()
            # 预测confidence大于0.5的地方*预测对的地方*目标的confidence
            detected_mask = conf50 * class_mask * tconf
            # iou大于0.5的且置信度大于0.5且预测对的地方，除以所有confidence大于0.5的数量
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            # iou大于0.5的且置信度大于0.5且预测对的地方，除以所有预测对的数量
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            # iou大于0.75的且置信度大于0.5且预测对的地方，除以所有预测对的数量
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }

            return output, total_loss


class Darknet(nn.Module):
    """YOLOv3 物体检测模型"""

    def __init__(self, config_path, img_size=416, loss_mode='baseline', reg_mode='baseline'):
        """
        arguments
        :param config_path: yolov3配置信息的位置
        :param img_size: 输出yolov3模型的图像尺寸
        :param loss_mode: loss的模式，[baseline, iou, giou, diou, ciou],默认是baseline，
        :param reg_mode: 正则化模式，[baseline, smooth, dropblock, all],默认是baseline
        """
        super(Darknet, self).__init__()
        # 将模型参数文件解析成一个列表，包含多个字典
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        # 将yolo_layers选出来
        self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")]
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)
        self.loss_mode = loss_mode
        self.reg_mode = reg_mode

    def forward(self, x, targets=None):
        # 原始图像的尺寸
        img_dim = x.shape[2]
        loss = 0
        layer_outputs, yolo_outputs = [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                # 正常卷积，池化，上采样，采用stride=2进行下采样
                x = module(x)
                if self.reg_mode == 'dropblock' or self.reg_mode == 'all':
                    x = DropBlock2D(0.3, 5)(x)
            elif module_def["type"] == "route":
                # route，将module_def["layers"]中的位置进行concat
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            elif module_def["type"] == "shortcut":
                # shortcut,将最后一次x和int(module_def["from"])位置的x相加
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                # 取module[0]是因为后面还有两个loss，
                x, layer_loss = module[0](x, targets, img_dim, self.loss_mode, self.reg_mode)
                loss += layer_loss
                # yolo共有三个尺度的结果，因此记录在一起
                yolo_outputs.append(x)
            # 记录每一次操作产生的x，便于shortcut和route取数
            layer_outputs.append(x)
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()

if __name__ == '__main__':
    # 测试2分类的yolov3_tiny_face_mask,
    cfg_file = './config/yolov3_tiny_face_mask.cfg'
    # cfg_file = './config/yolov3_face_mask.cfg'
    model = Darknet(cfg_file, loss_mode='diou')
    # print(model)
    torch.manual_seed(0)
    img = torch.rand((4,3,416,416))
    targets = torch.rand((8,6))
    targets[:,1] = torch.FloatTensor([1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0])
    result = model(img, targets)
    print(result[0])
    print(result[1].shape)



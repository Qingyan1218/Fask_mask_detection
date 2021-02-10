import torch
import numpy as np


def to_cpu(tensor):
    return tensor.detach().cpu()


def load_classes(path):
    """
    加载分类，返回一个列表
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def weights_init_normal(m):
    """初始化权重"""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def rescale_boxes(boxes, current_dim, original_shape):
    """ 将bbox缩放到原始图片大小
     原理：boxes是在current_dim下的坐标，即在一个current x current大小的框中，
     unpad_h和unpad_w即对应于current_dim下的图像坐标，即最大值等于current_dim，
     pad_x和pad_y是current_dim下原图和current_dim的差值，x-pad_x//2即表示x相对于
     unpad_w的坐标，(x-pad_x//2)/unpad_w * orig_w即缩放到原图大小的坐标"""
    # 原始图像大小
    orig_h, orig_w = original_shape
    # 如果图像的h大于w，则水平方向要pad第一个max等于h-w，第二个max等于h，即(h-w)*current_dim/h
    # 即宽度和高度的差值缩放到current_dim上
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # 未pad的图像大小，即unpad_h/unpad_w=orig_h/orig_w
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # 将bbox是current_dim下的/ unpad_w * orig_w即缩放到原图大小
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


def xywh2xyxy(x):
    """从(center x, center y, width, height) 变成(x1, y1, x2, y2)"""
    y = x.new(x.shape)
    # 中心点x减去w的一半即左上角x，余同
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ 计算average precision, given the recall and precision curves.
    # 参数
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # 返回
        和fasterrcnn中一样的average precision
    """

    # 根据物体的置信度排序
    i = np.argsort(-conf)
    # tp，conf和pred_cls根据置信度排序
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # 去除重复的分类
    unique_classes = np.unique(target_cls)

    # 创造pr曲线，计算每个类别AP
    ap, p, r = [], [], []
    for c in (unique_classes):
        # c是检测出的分类中的某一类，将所有预测为某一类的索引找出来
        i = pred_cls == c
        # 计算gt中属于c类的总数
        n_gt = (target_cls == c).sum()
        # 计算预测属于c类的总数
        n_p = i.sum()

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # 计算FPs和TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # 通过recall-precision curve计算AP
            ap.append(compute_ap(recall_curve, precision_curve))

    #计算F1 score
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # 加入两端的数据
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # 为了计算PR曲线下的面积，寻找x轴（recall）上改变值的
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_batch_statistics(outputs, targets, iou_threshold):
    """ 每个样本计算true positives, predicted scores和 predicted labels
    :param outputs：列表，每一项(x1, y1, x2, y2, object_conf, class_score, class_pred)
    :param targets：（0，category_id, x1，y1，x2，y2）
    """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue
        # 取出第i个样本，
        output = outputs[sample_i]
        # bbox
        pred_boxes = output[:, :4]
        # object_conf
        pred_scores = output[:, 4]
        # class_pred
        pred_labels = output[:, -1]

        # pred_boxes有n行
        true_positives = np.zeros(pred_boxes.shape[0])
        # targets[targets[:, 0] == sample_i]选出当前样本
        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        # 目标标签
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            # 用于记录已经找到的检测框
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # 如果检测到的框数量和gt一致，则结束循环
                if len(detected_boxes) == len(annotations):
                    break

                # 如果预测的标签不在gt标签内，跳出循环
                if pred_label not in target_labels:
                    continue
                # 计算每一个box和所有目标框最大的iou，并得到索引
                # 因为预测框的顺序和gt未必一致，因此要通过iou进行匹配
                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                # 如果iou超过阈值且找到的框没有被加入已找到的检测框列表
                if iou >= iou_threshold and box_index not in detected_boxes:
                    # tp列表中该框索引的位置置为1
                    true_positives[pred_i] = 1
                    # 将这个框加入已找到的检测框列表
                    detected_boxes += [box_index]
        # n代表box的数量，true_positives.shape=(n,) ,
        # pred_scores.shape=torch.Size([n]), pred_labels.shape)=torch.Size([n])
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics


def bbox_wh_iou(wh1, wh2):
    """通过宽和高计算iou，即认为中心点一致，用于计算anchor和groundtruth的iou
    wh1是3行2列，代表3个anchor的w和h，wh2是n行2列，代表n个gt_box的w和h"""
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    # 1行n列
    w2, h2 = wh2[0], wh2[1]
    # 找到w1和一行w2中的最小值，输出也是一行n列
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    返回两个bbos的iou
    """
    if not x1y1x2y2:
        # 从中心转变成角点
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # 获取bbox的四个角点
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # 获取交集的大小
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # 交集的大小
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # 并集的大小
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    :param: prediction,即模型输出的值；
    :param: conf_thres,小于置信度的框认为无效；
    :param: nms_thres,iou大于nms_thres的值认为是同一个框
    返回:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # 从(center x, center y, width, height) 变成(x1, y1, x2, y2)
    # prediction.shape: torch.Size([batchsize, 10647, 85])
    # [..., :4]不管前面的维度，只取最后一维的前四个数，即bbox
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    # 新建一个列表记录nms后的bbox
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # image_pred.shape:torch.Size([10647, 85]),筛选后变成torch.Size([n, 85])
        # 10647代表特征图上所有的像素点，每个点85维
        # 第4位表示是否是物体的confidence，过滤掉小于阈值的值，
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # 如果没有大于阈值的像素点，即未检测到物体，则转到下一张图片
        if not image_pred.size(0):
            continue
        # 物体的confidence（85维中的第5位）乘以分类的confidence（共80位）
        # image_pred[:, 5:].max(1)[0]，取出80个分类中最大的那个值
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # (-score).argsort()找出score从大到小排列的索引，然后将image_pred用该索引排序
        image_pred = image_pred[(-score).argsort()]
        # 取出后最后一维中80个类别的最大值及索引位置
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        # 将前4位的坐标及物体confidence，类别的confidence，类别的索引concat在一起
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # 开始nms，keep_boxes为最后保留的bbox
        keep_boxes = []
        while detections.size(0):
            # 将第0个框与所有框做iou，得到大于阈值的框，tensor([ True,  True, False, ...])
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            # 将第0个框的类别与所有框的类别作比较，一样的为True，找到与第0个框同一类的物体，
            label_match = detections[0, -1] == detections[:, -1]
            # 找出那些iou大于阈值且与第一个框类别一致的框，这些是无效的框，包括第0个框自己
            invalid = large_overlap & label_match
            # 将这些框的object_conf取出作为权重
            weights = detections[invalid, 4:5]
            # 然后将这些框的坐标根据权重进行融合
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            # 将第一个框加入保留框列表
            keep_boxes += [detections[0]]
            # 移除无效框，包括第0个框
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):
    """根据ground truth和预测值计算一系列数据"""
    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    # pred_boxes.shape=torch.Size([batchsize, 3, h/32, w/32, 4])->h/16, w.16->h/8, w/8
    # targets.shape = torch.Size([n, cls, cx, cy, w, h])
    # nB是batchsize
    nB = pred_boxes.size(0)
    # nA=3
    nA = pred_boxes.size(1)
    # nC=80，表示分类的数量
    nC = pred_cls.size(-1)
    # nG是当前特征图的大小，h/32, w/32->h/16, w.16->h/8, w/8
    nG = pred_boxes.size(2)

    # 初始化输出张量，torch.Size([batchsize, 3 , h/32 , w/32])->h/16, w/16->h/8, w/8
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
    # obj_mask代表有没有物体，class_mask代表类别，因为最后一个维度取消了，因此只能预测一个类别
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)

    # torch.Size([batchsize, 3 , h/32 , w/32, 80])->h/16, w/16->h/8, w/8
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    # 转成相对于box的坐标
    # 目标框=归一化后的坐标*特征图大小
    target_boxes = target[:, 2:6] * nG
    # gxy是gt的中心点坐标，gwh是gt的宽和高
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # 获取最好iou的box，ious是3行n列的矩阵，每一行是一个anchor和所有gt的iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    # 从3行中的每一列取最大值，并输出哪个位置的值最大，即和哪个anchor的iou最大，都是一行n列
    best_ious, best_n = ious.max(0)
    # b是targets的target[:, 0]变成一行，代表该batch中的第几个样本
    # target_labels是targets的target[:, 1]变成一行，
    b, target_labels = target[:, :2].long().t()
    # gx，gy，gw，gh是groundtruth的x，y，w，h，1行n列
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    # 格子的i和j就是x和y取整，1行n列
    gi, gj = gxy.long().t()
    # 设置掩码，b代表batchsize，即该batch中的第几个样本
    # best_n代表最好的框是第几个anchor产生的，
    # gj，gi代表图像的中心点所在的cell，该cell负责预测物体
    # 以上均是一行n列，即zip(b,best_n,gj,gi)组合出的n个位置处设置
    # obj_mask中best_n处的值为1，noobj_mask中best_n处的值为0，均代表有物体
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # iou超过ignore_threshold的地方noobj_mask设置为0，即认为有物体，ious.t是n行3列，
    # 如果一个bbox和3个anchor的iou分别是0.95，0.85，0.8，则obj_mask是1，0，0
    # 而noobj_mask是0，1，1，而后两个iou很大，代表是物体的概率很高，因此直接认为是负样本是不妥的
    # 因此设定ignore_thresh，将noobj_mask变成0，0，0，后面两个0表示这些有歧义的样本不参与计算
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # 坐标，gx.floor()即向下取整，有物体中心的格子存储坐标值
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # 宽度和高度，针对最好的那个anchor进行缩放,anchors是3行2列
    # 因此有了训练结果才知道需要针对哪个anchor进行缩放
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # 对gt的label进行one_hot编码，target_labes是1行n列，即80类中该类别位置的地方为1
    tcls[b, best_n, gj, gi, target_labels] = 1
    # pred_cls是预测值，torch.Size([batchsize, 3, h/32, w/32， 80])->h/16, w.16->h/8, w/8
    # 那些有物体中心的地方预测对类别的话掩码变成1，即最后一维中最大值所在的位置即类别
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    # 有物体中心的地方计算iou
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)
    # tconf即grount truth confidence，标签即obj_mask，有物体中心的地方为1
    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf


def build_iou_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):
    """根据ground truth和预测值计算一系列数据
    在build_targets基础上增加一个输出t_box，用来计算ciou，
    原来的x，y，w，h依旧输出，用来对比采用ciou后，坐标的回归"""
    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    # pred_boxes.shape=torch.Size([batchsize, 3, h/32, w/32, 4])->h/16, w.16->h/8, w/8
    # targets.shape = torch.Size([n, cls, cx, cy, w, h])
    # nB是batchsize
    nB = pred_boxes.size(0)
    # nA=3
    nA = pred_boxes.size(1)
    # nC=80，表示分类的数量
    nC = pred_cls.size(-1)
    # nG是当前特征图的大小，h/32, w/32->h/16, w.16->h/8, w/8
    nG = pred_boxes.size(2)

    # 初始化输出张量，torch.Size([batchsize, 3 , h/32 , w/32])->h/16, w/16->h/8, w/8
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
    # obj_mask代表有没有物体，class_mask代表类别，因为最后一个维度取消了，因此只能预测一个类别
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    # 增加一个矩阵用于记录中心点和宽高值
    t_box = FloatTensor(nB, nA, nG, nG, 4).fill_(0)

    # torch.Size([batchsize, 3 , h/32 , w/32, 80])->h/16, w/16->h/8, w/8
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    # 转成相对于box的坐标
    # 目标框=归一化后的坐标*特征图大小，因为是计算CIOU
    target_boxes = target[:, 2:6] * nG
    # gxy是gt的中心点坐标，gwh是gt的宽和高
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # 获取最好iou的box，ious是3行n列的矩阵，每一行是一个anchor和所有gt的iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    # 从3行中的每一列取最大值，并输出哪个位置的值最大，即和哪个anchor的iou最大，都是一行n列
    best_ious, best_n = ious.max(0)
    # b是targets的target[:, 0]变成一行，代表该batch中的第几个样本
    # target_labels是targets的target[:, 1]变成一行，
    b, target_labels = target[:, :2].long().t()
    # gx，gy，gw，gh是groundtruth的x，y，w，h，1行n列
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    # 格子的i和j就是x和y取整，1行n列
    gi, gj = gxy.long().t()
    # 设置掩码，b代表batchsize，即该batch中的第几个样本
    # best_n代表最好的框是第几个anchor产生的，
    # gj，gi代表图像的中心点所在的cell，该cell负责预测物体
    # 以上均是一行n列，即zip(b,best_n,gj,gi)组合出的n个位置处设置
    # obj_mask中best_n处的值为1，noobj_mask中best_n处的值为0，均代表有物体
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # iou超过ignore_threshold的地方noobj_mask设置为0，即认为有物体，ious.t是n行3列，
    # 如果一个bbox和3个anchor的iou分别是0.95，0.85，0.8，则obj_mask是1，0，0
    # 而noobj_mask是0，1，1，而后两个iou很大，代表是物体的概率很高，因此直接认为是负样本是不妥的
    # 因此设定ignore_thresh，将noobj_mask变成0，0，0，后面两个0表示这些有歧义的样本不参与计算
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # 坐标，有物体中心的格子存储坐标值，因为是CIOU，不需要归一化
    t_box[b, best_n, gj, gi, 0] = gx
    t_box[b, best_n, gj, gi, 1] = gy
    t_box[b, best_n, gj, gi, 2] = gw
    t_box[b, best_n, gj, gi, 3] = gh

    # 坐标，gx.floor()即向下取整，有物体中心的格子存储坐标值
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # 宽度和高度，针对最好的那个anchor进行缩放,anchors是3行2列
    # 因此有了训练结果才知道需要针对哪个anchor进行缩放
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # 对gt的label进行one_hot编码，target_labes是1行n列，即80类中该类别位置的地方为1
    tcls[b, best_n, gj, gi, target_labels] = 1
    # pred_cls是预测值，torch.Size([batchsize, 3, h/32, w/32， 80])->h/16, w.16->h/8, w/8
    # 那些有物体中心的地方预测对类别的话掩码变成1，即最后一维中最大值所在的位置即类别
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    # 有物体中心的地方计算iou
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)
    # tconf即grount truth confidence，标签即obj_mask，有物体中心的地方为1
    tconf = obj_mask.float()
    # 输出增加一个t_box
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf, t_box






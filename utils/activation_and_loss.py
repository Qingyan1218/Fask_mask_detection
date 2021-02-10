import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        # softplus = ln(1 + e^x)
        x = x * (torch.tanh(F.softplus(x)))
        return x


def iou_loss(boxes1, boxes2, iou_mode = 'ciou'):
    '''
    计算ciou = 1 - iou - p2/c2 - av，用ciou代替四个坐标的单独回归
    :param boxes1: 预测的结果，前面的维度与boxes2统一即可，最后一位是4就行
    :param boxes2: ground truth，与boxes1一致
    :return: 各种类型的iou
    '''

    # 变成左上角坐标、右下角坐标
    boxes1_x0y0x1y1 = torch.cat((boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                             boxes1[..., :2] + boxes1[..., 2:] * 0.5), dim=-1)
    boxes2_x0y0x1y1 = torch.cat((boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                             boxes2[..., :2] + boxes2[..., 2:] * 0.5), dim=-1)
    '''
    逐个位置比较boxes1_x0y0x1y1[..., :2]和boxes1_x0y0x1y1[..., 2:]，即逐个位置比较[x0, y0]和[x1, y1]，小的留下。
    比如留下了[x0, y0]
    这一步是为了避免一开始w h 是负数，导致x0y0成了右下角坐标，x1y1成了左上角坐标。
    '''
    boxes1_x0y0x1y1 = torch.cat((torch.min(boxes1_x0y0x1y1[..., :2], boxes1_x0y0x1y1[..., 2:]),
                             torch.max(boxes1_x0y0x1y1[..., :2], boxes1_x0y0x1y1[..., 2:])), dim=-1)
    boxes2_x0y0x1y1 = torch.cat((torch.min(boxes2_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., 2:]),
                             torch.max(boxes2_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., 2:])), dim=-1)

    # 两个矩形的面积
    boxes1_area = (boxes1_x0y0x1y1[..., 2] - boxes1_x0y0x1y1[..., 0]) * (
                boxes1_x0y0x1y1[..., 3] - boxes1_x0y0x1y1[..., 1])
    boxes2_area = (boxes2_x0y0x1y1[..., 2] - boxes2_x0y0x1y1[..., 0]) * (
                boxes2_x0y0x1y1[..., 3] - boxes2_x0y0x1y1[..., 1])

    # 相交矩形的左上角坐标、右下角坐标，shape 都是 (8, 13, 13, 3, 2)
    left_up = torch.max(boxes1_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., :2])
    right_down = torch.min(boxes1_x0y0x1y1[..., 2:], boxes2_x0y0x1y1[..., 2:])

    # 相交矩形的面积inter_area和union_area, 计算iou
    inter_section = right_down - left_up
    inter_section = torch.where(inter_section < 0.0, inter_section*0, inter_section)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / (union_area + 1e-9)

    # 包围矩形的左上角坐标、右下角坐标，shape 都是 (8, 13, 13, 3, 2)
    enclose_left_up = torch.min(boxes1_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., :2])
    enclose_right_down = torch.max(boxes1_x0y0x1y1[..., 2:], boxes2_x0y0x1y1[..., 2:])


    # 包围矩形的对角线的平方
    enclose_wh = enclose_right_down - enclose_left_up
    enclose_c2 = torch.pow(enclose_wh[..., 0], 2) + torch.pow(enclose_wh[..., 1], 2)
    # 包围矩形的面积
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]

    # 两矩形中心点距离的平方
    p2 = torch.pow(boxes1[..., 0] - boxes2[..., 0], 2) + torch.pow(boxes1[..., 1] - boxes2[..., 1], 2)
    # 增加av, 加上除0保护防止nan。
    atan1 = torch.atan(boxes1[..., 2] / (boxes1[..., 3] + 1e-9))
    atan2 = torch.atan(boxes2[..., 2] / (boxes2[..., 3] + 1e-9))
    v = 4.0 * torch.pow(atan1 - atan2, 2) / (math.pi ** 2)
    a = v / (1 - iou + v)

    if iou_mode == 'iou':
        iou = 1 - iou
        return iou
    elif iou_mode == 'giou':
        giou = 1 - iou + (enclose_area - union_area) / enclose_area
        return giou
    elif iou_mode == 'diou':
        diou = 1 - iou + 1.0 * p2 / enclose_c2
        return diou
    elif iou_mode == 'ciou':
        ciou = 1 - iou + 1.0 * p2 / enclose_c2 + 1.0 * a * v
        return ciou
    else:
        raise ValueError('no iou_mode named %s' % iou_mode)

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()

    def forward(self, x, target, smoothing=0.1):
        """
        arguments：
        :param x: x是模型的输出值，没有进行softmax
        :param target: labels，torch.long(),代表分类的一维向量
        :param smoothing: smoothing label的比率
        :return: 损失的平均值
        """
        confidence = 1. - smoothing
        # 先将x进行log_softmax，通过对比发现，F.cross_entropy总是采用F.log_softmax dim=1时的结果
        logprobs = F.log_softmax(x, dim=1)
        # print(F.nll_loss(logprobs, target))
        # print(F.cross_entropy(x, target))
        # torch.gather是在dim维上按照index中的顺序去找input中的数
        # t = torch.Tensor([[1,2],[3,4]]) torch.gather(t,1,torch.LongTensor([[0,0],[1,0]]))
        # 返回torch.Tensor([[1,1],[4,3]])
        # 即将最后一个维度对应分类处的值取出来
        nll_loss = -logprobs.gather(dim=1, index=target.unsqueeze(1))
        # print(nll_loss.mean())
        # 取消最后一个维度，由n行1列变成(n,)
        nll_loss = nll_loss.squeeze(1)
        # 取最后一维所有值的均值即ce_loss的值
        smooth_loss = -logprobs.mean(dim=-1)
        # 按照比例相加，最后求均值
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()

class DropBlock2D(nn.Module):
    def __init__(self, drop_prob, blcok_size):
        super(DropBlock2D, self).__init__()
        # drop_prob = 1 - keep_prob
        self.drop_prob = drop_prob
        # 一般是5或7
        self.block_size = blcok_size

    def forward(self, x):
        # print(x.dim)
        # assert x.dim == 4, \
        #     "Expected input with 4 dimensions(bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.0:
            return x
        else:
            # 此处用的是近似计算，
            gamma = self._compute_gamma(x)
            # sample mask，注意，没有为维度1
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()
            mask = mask.to(x.device)

            block_mask = self._compute_block_mask(mask)

            # 将掩码与x相乘，即每个channel上丢弃的部分乘以0
            out = x * block_mask[:, None, :, :]
            out = out * block_mask.numel() / block_mask.sum()
            return out

    def _compute_gamma(self, x):
        # 完整计算：accurate_gamma = drop_prob / block_size ** 2
        # *(x.shape[2]*x.shape[3])/((x.shape[2]-block_size+1)*(x.shape[3]-block_size+1))
        return self.drop_prob / (self.block_size ** 2)

    def _compute_block_mask(self, mask):
        # 采用max_pool的方式把中心为1的周边全部扩充为1，
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        # 如果无法整除，那么最后一个像素舍弃
        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        # 将block_mask 反转，原来是1的地方全部变成0
        block_mask = 1 - block_mask.squeeze(1)
        return block_mask


if __name__ == '__main__':
    # torch.manual_seed(1)
    # # yolov3的输出和target都是两个维度，即多少个预测框和类别的概率
    # pred = torch.rand(5,2)
    # # torch.argmax(target, dim=-1) = torch.Tensor([1, 1, 0, 0, 1]).long()
    # target = torch.Tensor([[0.3, 0.7], [0.2, 0.8], [0.9, 0.1], [0.7, 0.3], [0.3, 0.7]])
    # target = torch.argmax(target, dim=-1)
    # print(target)
    #
    # loss = LabelSmoothingCrossEntropy()
    # result = loss(pred, target, smoothing=0.1)
    # print(result)
    #
    # x = torch.rand(4, 3, 12, 12)
    # model = DropBlock2D(0.3, 5)
    # result = model(x)
    # print(result.shape)

    box1 = torch.FloatTensor([[15, 15, 10, 20]])
    box2 = torch.FloatTensor([[20, 20, 20, 10]])
    # box3 = torch.FloatTensor([[35, 35, 10, 10]])
    for mode in ['iou', 'giou', 'diou', 'ciou']:
        iou_result = iou_loss(box1, box2, iou_mode=mode)
        print(iou_result)




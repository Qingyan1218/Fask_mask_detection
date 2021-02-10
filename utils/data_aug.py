import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 数据可视化
def show_img(img, labels):
    """绘制图像和bbox
    :param img: 图像矩阵，np.array
    :param labels: normalized bbox，[cls,xc,yc,w,h]
    :return: None
    """
    fig, ax = plt.subplots(1)
    cmap = plt.get_cmap("tab20b")
    # print('img.shape', img.shape)
    h, w, _ = img.shape
    ax.imshow(img)
    for box in labels:
        # 坐标被归一化了，[cls, xc, yc, w, h]
        x1 = int((box[1] - box[3] / 2) * w)
        y1 = int((box[2] - box[4] / 2) * h)
        b_w = int(box[3] * w)
        b_h = int(box[4] * h)
        # print(x1, y1, b_w, b_h)
        bbox = patches.Rectangle((x1, y1), b_w, b_h, linewidth=2,
                                 edgecolor=cmap(0), facecolor="none")
        ax.add_patch(bbox)
    plt.show()

def cal_iou(box1, box2, w, h):
    """计算两个bbox的iou
    arguments:
    :param box1:第一个框的box，归一化后的尺寸
    :param box2:第二个框的box，归一化后的尺寸
    :return 两个bbox的iou,两个box都是归一化后的[xc, yc, w ,h]
    """
    # 计算iou时需要还原到原图大小，否则iou结果不对
    b1_x1 = (box1[0] - box1[2] / 2) * w
    b1_y1 = (box1[1] - box1[3] / 2) * h
    b1_x2 = b1_x1 + box1[2] * w
    b1_y2 = b1_y1 + box1[3] * h

    b2_x1 = (box2[0] - box2[2] / 2) * w
    b2_y1 = (box2[1] - box2[3] / 2) * h
    b2_x2 = b2_x1 + box2[2] * w
    b2_y2 = b2_y1 + box2[3] * h

    # 获取交集的大小
    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)
    # 交集的大小
    inter_area = max(inter_rect_x2 - inter_rect_x1 + 1, 0) * max(
        inter_rect_y2 - inter_rect_y1 + 1, 0
    )
    # 并集的大小
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def check_box(box1, box2):
    """排除一个框包含另一个框，如果包含，返回True
    arguments:
    :param box1:归一化的第一个框
    :param box2:归一化的第二个框
    :return True or False
    """
    # 仅仅比较位置，所以不需要还原到原大小
    b1_x1 = box1[0] - box1[2] / 2
    b1_y1 = box1[1] - box1[3] / 2
    b1_x2 = b1_x1 + box1[2]
    b1_y2 = b1_y1 + box1[3]
    b2_x1 = box2[0] - box2[2] / 2
    b2_y1 = box2[1] - box2[3] / 2
    b2_x2 = b2_x1 + box2[2]
    b2_y2 = b2_y1 + box2[3]
    if b1_x1 <= b2_x1 and b1_y1 <= b2_y1 and b1_x2 >= b2_x2 and b1_y2 >= b2_y2:
        return True
    if b1_x1 >= b2_x1 and b1_y1 >= b2_y1 and b1_x2 <= b2_x2 and b1_y2 <= b2_y2:
        return True
    return False

def get_box(mix_labels, w, h, mix_ori_w,mix_ori_h):
    """从另一幅图中随机选择一个bbox的图像填充至当前图像"""
    # 初始化mix_w 和 mix_h，让while循环进行
    delta_w = 0
    delta_h = 0

    while delta_w <= 0 or delta_h <= 0:
        # print('image.shape', w, h)
        # 选出一个bbox作为填充，并计算实际中心及大小
        mix_box = mix_labels[random.randint(0, len(mix_labels) - 1)]
        mix_cls = mix_box[0]
        # 计算mix对象在新图中的位置
        mix_xc = int(mix_box[1] * mix_ori_w)
        mix_yc = int(mix_box[2] * mix_ori_h)
        mix_w = int(mix_box[3] * mix_ori_w)
        mix_h = int(mix_box[4] * mix_ori_h)
        delta_w = w - mix_w -1
        delta_h = h - mix_h -1
        # print('box.shape',mix_w,mix_h)
        # print('delta',delta_w, delta_h)

    # 随机选择一个左上角x, y，范围要保证填充进去的框在新的图像内
    x1_in_img = random.randint(0, w - mix_w -1)
    y1_in_img = random.randint(0, h - mix_h -1)
    xc_in_img = (x1_in_img + mix_w / 2) / w
    yc_in_img = (y1_in_img + mix_h / 2) / h
    w_in_img = mix_w / w
    h_in_img = mix_h / h

    # 计算mix对象在原图中的位置
    x1_in_mix_img = int(mix_xc - mix_w / 2)
    y1_in_mix_img = int(mix_yc - mix_h / 2)
    x2_in_mix_img = x1_in_mix_img + mix_w
    y2_in_min_img = y1_in_mix_img + mix_h

    # 返回mix对象在新图中的位置列表,归一化后的中心和宽高，及在原图中的位置列表,未归一化
    return [mix_cls, xc_in_img, yc_in_img, w_in_img, h_in_img],\
           [x1_in_mix_img,y1_in_mix_img,x2_in_mix_img,y2_in_min_img]

''' cutmix'''

def cut_mix(image_files, label_files, index):
    """将另一幅图中的某个bbox cut后mix到当前图中
    arguments
    :param image_files: 样本的路径列表
    :param label_files: 样本的标签路径列表
    :param index: 索引，用于self.__getitem__
    :return: img,labels,img是np.array,labels[cls,xc,yc,w,h]
    """
    # 首先打开index位置的图片作为背景
    img_path = image_files[index].rstrip()
    img = np.array((Image.open(img_path).convert('RGB')))
    if len(img.shape) != 3:
        # 灰度图先增加一个维度
        img = img[:,:,np.newaxis]
        # 然后扩展成三个channel
        img = np.concatenate([img,img,img],axis = 0)
    h, w, _ = img.shape
    # Labels
    label_path = label_files[index].rstrip()
    labels = np.loadtxt(label_path).reshape(-1, 5)

    # 根据论文，两张图片足够了，因此再随机选择一张图片的序号
    random_idx = random.randint(0, len(label_files) - 1)
    mix_img_path = image_files[random_idx].rstrip()
    mix_pic = (Image.open(mix_img_path).convert('RGB'))
    mix_w, mix_h = mix_pic.size
    mix_label_path = label_files[random_idx].rstrip()
    mix_labels = np.loadtxt(mix_label_path).reshape(-1, 5)
    # 为了让mix对象尽可能地在img中，缩放mix对象所在的图像大小
    if mix_h <= h and mix_w <= w:
        mix_img = np.array(mix_pic)
    else:
        # 选择比例小的，保证缩放后的图片一定小于img
        scale = min(h/mix_h, w/mix_w)
        mix_w = int(mix_w * scale)
        mix_h = int(mix_h * scale)
        mix_img = np.array(mix_pic.resize((mix_w, mix_h)))

    # 当mix对象和现有的bbox重叠严重时，舍弃
    # 设置循环标志
    iou_is_bad = True
    check_1 = 0
    check_2 = 0
    # 循环次数，如果找不到合适的mix对象，则放弃
    num = 0
    while num < 10 and iou_is_bad:
        # 如果能获得框，则进行判断，不符合就继续获取mix对象
        result = get_box(mix_labels, w, h, mix_w, mix_h)
        box_to_fill = result[0]
        box_in_mix_img = result[1]
        for bbox in labels:
            iou = cal_iou(box_to_fill[1:5], bbox[1:5], w, h)
            check_result = check_box(box_to_fill[1:5], bbox[1:5])
            if iou > 0.5:
                # 检查bbox和mix对象是否iou过大
                check_1 += 1
            if check_result == True:
                # 检查bbox和mix对象是否互相包含
                check_2 += 1
        # 如果mix对象符合要求，则停止循环，否则继续循环
        if check_1 == 0 and check_2 == 0:
            iou_is_bad = False
        else:
            iou_is_bad = True
        num += 1

    # 认为num=9时的mix对象已经是不满足要求的，那么就返回原来的img和labels
    if num < 9:
        # 计算mix对象在原图中的位置
        x1_in_mix_img = box_in_mix_img[0]
        y1_in_mix_img = box_in_mix_img[1]
        x2_in_mix_img = box_in_mix_img[2]
        y2_in_mix_img = box_in_mix_img[3]

        # 计算mix对象在新图中的位置，确保两个图中box尺寸一致
        x1 = int(box_to_fill[1] * w - (x2_in_mix_img - x1_in_mix_img) / 2)
        y1 = int(box_to_fill[2] * h - (y2_in_mix_img - y1_in_mix_img) / 2)
        x2 = x1 + x2_in_mix_img - x1_in_mix_img
        y2 = y1 + y2_in_mix_img - y1_in_mix_img

        # img中增加mix对象
        img[y1:y2, x1:x2, :] = mix_img[y1_in_mix_img:y2_in_mix_img,
                               x1_in_mix_img:x2_in_mix_img, :]

        labels = np.concatenate((labels, np.array(box_to_fill).reshape(1,5)), axis=0)
    # show_img(img, labels)
    # 返回的labels是归一化的[cls,xc,yc,w,h]格式
    return img, labels


''' Mosaic'''
def mosaic(img_size, image_files, label_files, index):
    """
    arguments:
    :param img_size: 训练图像用的尺寸，放大两倍后填充4张图片，再缩放回img_size
    :param image_files: 样本的路径列表
    :param label_files: 样本的标签路径列表
    :param index: 索引，用于self.__getitem__
    :return: img,labels,img是np.array,labels[cls,xc,yc,w,h]
    """
    # 用于记录拼图中的labels
    labels4 = []
    # s是训练图像用的尺寸
    s = img_size
    # 随机选择一个中心x, y，范围从0.5s到1.5s
    xc, yc = [int(random.uniform(s * 0.9, s * 1.1)) for _ in range(2)]
    # 从所有的图片当中选择其他的3张图片
    indices = [index] + [random.randint(0, len(label_files) - 1) for _ in range(3)]

    for i, idx in enumerate(indices):
        # 循环4次，用4张图片填充
        # 获取图像矩阵、高和宽
        img_path = image_files[idx].rstrip()
        img = np.array((Image.open(img_path).convert('RGB')))
        if len(img.shape) != 3:
            # 灰度图先增加一个维度
            img = img[:, :, np.newaxis]
            # 然后扩展成三个channel
            img = np.concatenate([img, img, img], axis=0)
        h,w,_ = img.shape

        # 以下代码中，下标带a的表示在img4上填充位置的左上角和右下角坐标
        # 下标带b的表示原图中切割出来部分的左上角和右下角坐标
        # min和max部分分别表示不能超出所在图像的边界
        # 对于左上角，原图的右下角一定会放入img4，余同
        if i == 0:
            # 把新图像先设置成原来的4倍，到时候再resize回去，114是gray
            # 不能在其他地方初始化img4，循环外的话没有img，if外的话每次都会覆盖
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        # 右上角，原图的左下角一定会放入img4
        elif i == 1:
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        # 左下角，原图的右上角一定会放入img4
        elif i == 2:
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        # 右下角，原图的左上角一定会放入img4
        elif i == 3:
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        # 新的图像用原来的图像切割出来的部分填充
        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
        # 原图填充到img4后，相应的bbox也需要偏移
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        label_path = label_files[idx].rstrip()
        raw_labels = np.loadtxt(label_path).reshape(-1, 5)
        labels = raw_labels.copy()
        # 将中心和宽高格式xywh变成左上右下xyxy格式
        if len(raw_labels) > 0:
            # 此时x是0-1，同时，label是[class, bbox_xc, bbox_yc, bbox_w, bbox_c]
            labels[:, 1] = w * (raw_labels[:, 1] - raw_labels[:, 3] / 2) + padw
            labels[:, 2] = h * (raw_labels[:, 2] - raw_labels[:, 4] / 2) + padh
            labels[:, 3] = w * (raw_labels[:, 1] + raw_labels[:, 3] / 2) + padw
            labels[:, 4] = h * (raw_labels[:, 2] + raw_labels[:, 4] / 2) + padh
        # 将4个图片的label全部加入
        labels4.append(labels)

    if len(labels4):
        # 沿着第0位concat
        labels4 = np.concatenate(labels4, 0)
        # 随机剪裁，bbox的坐标裁剪到0~416*2范围内，即img4范围内
        # 还有一种是中心裁剪，将bbox坐标减去416/2后裁剪到0~s，结果存到labels中
        # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])

    # 一种是中心剪裁，即将中心一块区域裁剪出来，这个还需要box的剪裁
    # img4 = img4[s // 2: int(s * 1.5), s // 2:int(s * 1.5)]
    # 还有一种是随机剪裁，同时可以进行仿射投影变换等，bbox相应地变换
    # 这里不进行裁剪，直接由datasets进行缩放，
    # bbox剪裁之后不都是有效的，因此需要筛选，将box宽和高等于0的地方全部删除
    valid_label = []
    for box in labels4:
        cls = box[0]
        b_w = int(box[3] - box[1])
        b_h = int(box[4] - box[2])
        # 如果bbox的宽或高等于0，则不加入valid_label
        if b_w != 0 and b_h != 0:
            xc = (box[1] + box[3]) / (4 * s)
            yc = (box[2] + box[4]) / (4 * s)
            b_w = b_w / (2 * s)
            b_h = b_h / (2 * s)
            valid_label.append([cls, xc, yc, b_w, b_h])
    if valid_label:
        # 如果合并后存在有效的框
        # show_img(img4, labels4)
        # 返回的labels是归一化的[cls,xc,yc,w,h]格式
        labels4 = np.array(valid_label)
        return img4, labels4
    else:
        # 否则返回原图和原来的labels
        return img, raw_labels








import numpy as np
import torch.onnx
from models import Darknet
from torch.utils.data import DataLoader
from utils.datasets import ImageFolder
from utils.utils import non_max_suppression, rescale_boxes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from PIL import Image
import random

import onnxruntime
import onnx

# 加载预训练模型
torch_model = Darknet('./config/yolov3_tiny_face_mask.cfg')
model_check = './checkpoints/yolov3_ckpt_50_tiny_face_mask.pth'
torch_model.load_state_dict(torch.load(model_check, map_location=torch.device('cpu')))
# 切换模型为inference
torch_model.eval()

image_size = 416
# x设定为一个和模型的输入具有相同尺寸的随机tensor
x = torch.rand(4, 3, image_size, image_size)
"""
参数1：模型
参数2：输入，只要形状对即可
参数3：模型保存的地方
export_params:store the trained parameter weights inside the model file
opset_version:the ONNX version to export the model to
do_constant_folding:whether to execute constant folding for optimization
input_names:the model's input names
output_names:the model's output names
dinamic_axes:variable lenght axes
"""
torch.onnx.export(torch_model, x, "super_resolution.onnx",
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'},
                                'output': {0: 'batch_size'}})

onnx_model = onnx.load("super_resolution.onnx")
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession("super_resolution.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# 以下是进行前向推理
# 加载图片数据，拿一个batch的图片做测试
img_data = ImageFolder('./test_image', img_size=image_size)
dataloader = DataLoader(img_data, batch_size=4,shuffle=False)
imgs_path = []
img_detections = []
conf_thresh = 0.8
# 根据实验，nms_thres设为0.3能够有效去掉重复的框
nms_thresh = 0.3

for i,(path,img) in enumerate(dataloader):
    x = img
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
    # 将结果转变为tensor，shape为torch.Size([1, 4, 10647, 7])，因此去掉第0维
    detections = torch.FloatTensor(ort_outs).squeeze(0)
    detections = non_max_suppression(detections, conf_thresh, nms_thresh)
    imgs_path.extend(path)
    img_detections.extend(detections)

classes = ['face', 'face_mask']

def show_bbox(path, detections, save_img=False):
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    # 打开原图
    img = np.array(Image.open(path))
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # 绘制边框和类别
    if detections is not None:
        # 将bbox缩放到原图大小
        detections = rescale_boxes(detections, image_size, img.shape[:2])
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

            print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

            box_w = x2 - x1
            box_h = y2 - y1

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            # 绘制一个矩形
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
            # Add the bbox to the plot
            ax.add_patch(bbox)
            # Add label
            plt.text(
                x1,
                y1,
                s=classes[int(cls_pred)],
                color="white",
                verticalalignment="top",
                bbox={"color": color, "pad": 0},
            )

    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    if save_img:
        filename = path.split("/")[-1].split(".")[0]
        plt.savefig(f"onnx_output/{filename}.png", bbox_inches="tight", pad_inches=0.0)
    else:
        plt.show()

for img_i, (path, detections) in enumerate(zip(imgs_path, img_detections)):
    print("(%d) Image: '%s'" % (img_i, path))
    show_bbox(path, detections)
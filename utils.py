import os
import json
import pickle
import random
import numpy as np
from matplotlib import colors
from dice_coefficient_loss import *
from model.MixUNet import MixUNet
import matplotlib.pyplot as plt


def model_selection(model_name: str, num_class: int, in_channel: int, patch_size: int, drop_rate, device):
    if model_name == "mixunet":
        net = MixUNet(in_channels=in_channel, num_classes=num_class,
                      bilinear=False,
                      base_c=64,
                      patch_size=patch_size,
                      depths=(2, 2, 4, 2),
                      num_heads=(2, 4, 8, 16),
                      window_size=4,
                      dwconv_kernel_size=3,
                      mlp_ratio=4.,
                      qkv_bias=True,
                      qk_scale=None,
                      norm_layer=nn.LayerNorm,
                      val1=False,
                      val2=False).to(device)
    else:
        net = None
        assert "no search such model"
    return net


def read_data(root: str, random_seed: int = 0):
    random.seed(random_seed)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    images_path = []  # 存储验证集的所有图片路径
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG", ".mat"]  # 支持的文件后缀类型

    images = [os.path.join(root, i) for i in os.listdir(root)
              if os.path.splitext(i)[-1] in supported]
    # 排序，保证各平台顺序一致
    images.sort()
    # 记录该类别的样本数量
    every_class_num.append(len(images))

    for img_path in images:
        images_path.append(img_path)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    assert len(images_path) > 0, "number of training images must greater than 0."

    return images_path


def plot_confusion_matrix(conf_matrix, save_path, labels_names=None):
    """
    conf_matrix: 输入的混淆矩阵，为numpy型的Array
    labels_names: 类别名。
    save_path: 图片保存路径
    """
    # 绘制混淆矩阵
    if labels_names is None:
        labels_names = ['Control', 'Tumor']
    classes = conf_matrix.shape[0]
    labels_names = ['Control', 'Tumor']  # 每种类别的标签

    # 创建一个颜色矩阵，用于自定义颜色
    colors_matrix = np.full(conf_matrix.shape, 0.0)  # 默认为白色（值为1）
    np.fill_diagonal(colors_matrix, 1.0)  # 主对角线设为0，用于天蓝色

    # 需要将RGB值转化为[0, 1]的范围
    deep_skyblue_rgb = [0 / 255, 75 / 255, 126 / 255]
    white = [229 / 255, 240 / 255, 249 / 255]
    cmap = colors.LinearSegmentedColormap.from_list("", [white, deep_skyblue_rgb])

    # 创建绘图对象
    fig, ax = plt.subplots()

    # 用自定义颜色矩阵绘制图像，使用 `cmap` 为天蓝色渐变
    cax = ax.imshow(colors_matrix, cmap=cmap, interpolation='nearest')

    for x in range(classes):
        for y in range(classes):
            # 注意这里的matrix[y, x]不是matrix[x, y]
            info = int(conf_matrix[y, x])
            # 判断是否在对角线上
            if x == y:
                if x == 1:
                    color = "white"  # "tab:red"
                else:
                    color = "white"
            else:
                color = "black"
            plt.text(x, y, info,
                     verticalalignment='center',
                     horizontalalignment='center',
                     fontdict={'family': 'Times New Roman', 'weight': 'bold', 'size': 18},
                     color=color)

    # 设置x轴和y轴标签
    ax.set_xticks(range(classes))
    ax.set_xticklabels(labels_names, fontdict={'family': 'Times New Roman', 'weight': 'bold', 'size': 16})
    ax.set_yticks(range(classes))
    ax.set_yticklabels(labels_names, fontdict={'family': 'Times New Roman', 'weight': 'bold', 'size': 16})

    # 调整y轴刻度文字旋转90度，并贴近y轴
    ax.tick_params(axis='y', pad=5)  # 调整与y轴的距离
    for label in ax.get_yticklabels():
        label.set_rotation(90)  # 设置旋转90度
        label.set_verticalalignment('center')  # 垂直居中
        label.set_horizontalalignment('right')  # 水平右对齐

    ax.set_ylabel('True label', fontdict={'family': 'Times New Roman', 'weight': 'bold', 'size': 16})
    ax.set_xlabel('Predicted label', fontdict={'family': 'Times New Roman', 'weight': 'bold', 'size': 16})
    plt.title("Confusion Matrix", fontdict={'family': 'Times New Roman', 'weight': 'bold', 'size': 20})

    plt.tight_layout()  # 自动调整布局以避免重叠
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 保存为 PNG 格式，300 DPI 分辨率
    # 显示图像
    plt.show()

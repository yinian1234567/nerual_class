import torch
import cv2
import os
from setuptools.sandbox import save_path
from torchgen.executorch.api.et_cpp import return_names
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from transformers.image_utils import pil_torch_interpolation_mapping
from wx.py.sliceshell import outputStartText


# path = r"D:\File\PycharmProject\NeuroScience\images\0\156.png"

# 返回图像信息量权重
def dino(path):
    model_name = "facebook/dino-vitb16"
    model = ViTForImageClassification.from_pretrained(model_name, output_attentions=True)
    processor = ViTImageProcessor.from_pretrained(model_name)

    print(model)

    # 读取并预处理图像
    image_path = path  # 替换为你的图像路径
    image = Image.open(image_path).convert("RGB")
    image.copy()
    inputs = processor(images=image, return_tensors="pt")

    # 获取模型的输出（包括 Attention）
    with torch.no_grad():
        outputs = model(**inputs)

    # 提取 Self-Attention 权重
    # 取最后一层的 Attention Map (shape: [batch_size, num_heads, num_patches+1, num_patches+1])
    attentions = outputs.attentions[-1]  # 最后一层
    attentions = attentions.mean(dim=1).squeeze(0)  # 对多头进行平均，shape: [num_patches+1, num_patches+1]

    # 提取分类 Token 对所有 Patch 的 Attention 权重
    cls_attention = attentions[0, 1:]  # 分类 Token 对其他 Patch 的注意力
    cls_attention = cls_attention.reshape(14, 14).cpu().numpy()  # Reshape 为 14x14 对应的 Patch 网格

    # 归一化 Attention 权重
    cls_attention = (cls_attention - cls_attention.min()) / (cls_attention.max() - cls_attention.min())

    # 可视化热力图叠加到原图上
    def visualize_attention(image, attention, patch_size=16):
        # Resize attention map to match the original image size
        attention = Image.fromarray((attention * 255).astype(np.uint8)).resize(
            (image.size[0], image.size[1]), Image.NEAREST
        )
        attention = np.array(attention)

        # Plot the original image and attention heatmap
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(image)
        ax.imshow(attention, cmap="jet", alpha=0.5)  # 将热力图叠加到原图上
        ax.axis("off")
        plt.show()

    # 调用可视化函数
    visualize_attention(image, cls_attention)
    return cls_attention


def rank_matrix(input_array):
    """
    对一个 n x n 的 numpy 数组的元素值进行排序，返回每个元素的排名矩阵。

    参数:
    input_array (numpy.ndarray): 输入的 n x n numpy 数组，元素为 float 类型。

    返回:
    numpy.ndarray: 一个 n x n 的排名矩阵，每个元素代表对应输入元素的排名。
    """
    # 将数组展平并排序，获取排序索引
    flat_array = input_array.flatten()
    sorted_indices = np.argsort(flat_array)  # 升序排序的索引

    # 创建排名矩阵 (从 1 开始排名)
    ranks = np.zeros_like(flat_array, dtype=int)
    ranks[sorted_indices] = np.arange(1, len(flat_array) + 1)

    # 将排名矩阵恢复为 n x n 形状
    rank_matrix = ranks.reshape(input_array.shape)
    print(rank_matrix.shape)
    return rank_matrix


def max_pooling_2d(input_array,save_path):
    """
    对输入的二维数组执行最大池化操作，使宽和高都变为原来的一半。

    参数:
    - input_array (numpy.ndarray): 输入的二维数组，形状为 (H, W)。

    返回:
    - pooled_array (numpy.ndarray): 最大池化后的二维数组，形状为 (H//2, W//2)。
    """
    if input_array.shape[0] % 2 != 0 or input_array.shape[1] % 2 != 0:
        raise ValueError("Input array dimensions must be even for this operation.")

    # 获取输入数组的高度和宽度
    h, w = input_array.shape

    # 初始化输出数组 (H//2, W//2)
    pooled_array = np.zeros((h // 2, w // 2), dtype=input_array.dtype)

    # 遍历 2x2 窗口并取最大值
    for i in range(0, h, 2):
        for j in range(0, w, 2):
            pooled_array[i // 2, j // 2] = np.max(input_array[i:i + 2, j:j + 2])

    np.savetxt(save_path, pooled_array, fmt="%.5f")
    return pooled_array


#
# def display_image_in_blocks_by_rank(path, h, w, rank_array, fps):
#     """
#     按照 rank_array 的顺序展示图像块。
#
#     参数:
#     - path (str): 图像路径。
#     - h, w (int): 分块的行数和列数。
#     - rank_array (numpy.ndarray): 定义展示顺序的排名数组。
#     """
#     # 读取图像
#     img = cv2.imread(path)
#     if img is None:
#         raise ValueError(f"Image not found or unable to read: {path}")
#
#     # 获取图像的高度和宽度
#     height, width = img.shape[:2]
#
#     # 计算每个小块的高度和宽度
#     block_height = (height + h - 1) // h  # 向上取整
#     block_width = (width + w - 1) // w
#
#     # 创建一个全黑的图像，用于显示
#     black_img = np.zeros_like(img)
#
#     # 获取块的位置列表和展示顺序
#     block_positions = [(i, j) for i in range(h) for j in range(w)]
#     sorted_positions = sorted(block_positions, key=lambda pos: rank_array[pos[0], pos[1]])
#
#     # 按 rank_array 的顺序逐步显示小块
#     for i, j in sorted_positions:
#         y_start = i * block_height
#         y_end = min((i + 1) * block_height, height)
#         x_start = j * block_width
#         x_end = min((j + 1) * block_width, width)
#
#         # 将当前小块复制到全黑图像的相应位置
#         black_img[y_start:y_end, x_start:x_end] = img[y_start:y_end, x_start:x_end]
#
#         # 显示图像
#         cv2.imshow('Image in Blocks by Rank', black_img)
#         cv2.waitKey(int(1000/fps))  # 每显示一个小块暂停 500 毫秒
#
#     # 关闭所有窗口
#     cv2.destroyAllWindows()


def display_image_in_blocks_by_rank(path, h, w, rank_array, fps, save_dir=None):
    """
    按照 rank_array 的顺序展示图像块，并支持导出视频。

    参数:
    - path (str): 图像路径。
    - h, w (int): 分块的行数和列数。
    - rank_array (numpy.ndarray): 定义展示顺序的排名数组。
    - fps (int): 每秒帧数。
    - save_dir (str): 保存视频的目录。如果为 None，则存储到当前目录。
    """
    # 读取图像
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Image not found or unable to read: {path}")

    # 获取图像的高度和宽度
    height, width = img.shape[:2]

    # 计算每个小块的高度和宽度
    block_height = (height + h - 1) // h  # 向上取整
    block_width = (width + w - 1) // w

    # 创建一个全黑的图像，用于显示
    black_img = np.zeros_like(img)

    # 获取块的位置列表和展示顺序
    block_positions = [(i, j) for i in range(h) for j in range(w)]
    sorted_positions = sorted(block_positions, key=lambda pos: rank_array[pos[0], pos[1]])

    # 动态生成视频文件名
    video_filename = "vision.mp4"
    if save_dir is None:
        save_path = video_filename
    else:
        save_path = os.path.join(save_dir, video_filename)

    # 初始化视频写入对象
    video_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)
    )

    # 按 rank_array 的顺序逐步显示小块
    for i, j in sorted_positions:
        y_start = i * block_height
        y_end = min((i + 1) * block_height, height)
        x_start = j * block_width
        x_end = min((j + 1) * block_width, width)

        # 将当前小块复制到全黑图像的相应位置
        black_img[y_start:y_end, x_start:x_end] = img[y_start:y_end, x_start:x_end]

        # 写入视频帧
        video_writer.write(black_img)

        # 显示图像
        cv2.imshow('Image in Blocks by Rank', black_img)
        cv2.waitKey(int(1000 / fps))  # 暂停每帧的显示时间

    # 释放视频写入对象
    video_writer.release()

    # 关闭所有窗口
    cv2.destroyAllWindows()
    print(f"Video saved as: {save_path}")


def generate(data_path, fps, save_path):
    input_array = dino(data_path)
    input_array = max_pooling_2d(input_array,matrix_save_path)
    # result = rank_matrix(input_array)
    # display_image_in_blocks_by_rank(data_path, 7, 7, result, fps, save_path)


data_path = r"D:\File\PycharmProject\NeuroScience\images\1\horse\horse.png"
# 测试代码
fps = 2
matrix_save_path =r"D:\File\PycharmProject\NeuroScience\vision_stim\1\horse\weight_matrix.txt"
save_path = r"D:\File\PycharmProject\NeuroScience\vision_stim\2\horse"
generate(data_path, fps, save_path)

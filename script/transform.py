from PIL import Image
import os

n = 0

crop_width = 612  # 裁剪宽度
crop_height = 360  # 裁剪高度

image_folder = r"D:\File\PycharmProject\NeuroScience\images\cat"


def check():
    images = [os.path.join(image_folder, img) for img in os.listdir(image_folder)]
    for img_path in images:
        with Image.open(img_path) as img:
            w, h = img.size
            if w != crop_width or h != crop_height:
                print("error!")


def crop_center(image, crop_width, crop_height):
    width, height = image.size
    # 计算中心区域的坐标
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = (width + crop_width) // 2
    bottom = (height + crop_height) // 2
    return image.crop((left, top, right, bottom))


def crop_images_in_place(folder, crop_width, crop_height):
    # 遍历文件夹中的所有图片文件
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)

        # 打开图片、裁剪并保存
        with Image.open(file_path) as img:
            cropped_img = crop_center(img, crop_width, crop_height)
            cropped_img.save(file_path)  # 覆盖原始图片文件


def ensure_rgb(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        with Image.open(file_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
                img.save(file_path)  # 覆盖保存


def stretch_to_square(image_path, output_path, size):
    """
    将图片拉伸为指定大小的正方形。
    :param image_path: 输入图片路径
    :param output_path: 输出图片路径
    :param size: 目标正方形边长
    """
    with Image.open(image_path) as img:
        # 调整图片大小为正方形
        resized_img = img.resize((size, size), Image.LANCZOS)

        # 保存图片
        resized_img.save(output_path)

image_path=r"D:\File\PycharmProject\NeuroScience\images\2\dog\095.png"
output_path=r"D:\File\PycharmProject\NeuroScience\images\2\dog\095.png"

# 示例
stretch_to_square(image_path, output_path, 360)

# folder = "path_to_your_images_folder"  # 图片文件夹路径
# ensure_rgb(image_folder)
# crop_images_in_place(image_folder, crop_width, crop_height)

# (612,360)

# (400,267)

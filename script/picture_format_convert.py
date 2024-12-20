import os
from PIL import Image
from setuptools.sandbox import save_path


def convert_webp_to_png(input_folder, output_folder):
    # 创建输出文件夹，如果不存在的话
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 只处理 .webp 文件
        if filename.lower().endswith(".webp"):
            # 定义输入输出路径
            input_path = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(filename)[0] + ".png"
            output_path = os.path.join(output_folder, output_filename)
            # 打开 WebP 图像并转换为 PNG 格式
            with Image.open(input_path) as img:
                img.save(output_path, "PNG")
                print(f"Converted {filename} to PNG.")
            os.remove(input_path)


def convert_jepg_to_png(input_folder, output_folder):
    # 创建输出文件夹，如果不存在的话
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 只处理 .webp 文件
        if filename.lower().endswith(".jepg"):
            # 定义输入输出路径
            input_path = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(filename)[0] + ".png"
            output_path = os.path.join(output_folder, output_filename)
            # 打开 WebP 图像并转换为 PNG 格式
            with Image.open(input_path) as img:
                img.save(output_path, "PNG")
                print(f"Converted {filename} to PNG.")
            os.remove(input_path)

def delete_webp_images(folder):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder):
        # 检查文件后缀是否为 .webp（不区分大小写）
        if filename.lower().endswith(".webp"):
            file_path = os.path.join(folder, filename)
            try:
                os.remove(file_path)  # 删除文件
                print(f"Deleted: {filename}")
            except Exception as e:
                print(f"Failed to delete {filename}: {e}")


def rename(folder):
    files = os.listdir(folder)
    for i, filename in enumerate(files):
        file_ext = os.path.splitext(filename)[1]
        new_name = f"{i:03}{file_ext}"
        old_file = os.path.join(folder, filename)
        new_file = os.path.join(folder, new_name)
        os.rename(old_file, new_file)


data_path = r"D:\File\PycharmProject\NeuroScience\images\bird"
# rename(data_path)
convert_webp_to_png(data_path, data_path)
# delete_webp_images(output_path)

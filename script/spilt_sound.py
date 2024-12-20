from moviepy.editor import AudioFileClip
import os


def split_audio_of_mp4_files_in_folder(folder_path, output_dir):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件是否为MP4格式
        if filename.endswith(".mp4")or filename.endswith(".wav"):
            # 构建完整的输入文件路径
            input_file_path = os.path.join(folder_path, filename)

            # 获取不带扩展名的文件名作为base_name
            base_name = os.path.splitext(filename)[0]

            # 加载音频文件
            audio = AudioFileClip(input_file_path)

            # 获取音频总时长（秒）
            duration = audio.duration

            # 计算需要分割的段数，向上取整
            num_segments = int(duration // 3)

            # 遍历每一段，并保存
            for i in range(num_segments):
                start_time = i * 3
                end_time = start_time + 3

                # 获取子剪辑
                segment = audio.subclip(start_time, end_time)

                # 生成文件名
                segment_filename = f"{base_name}_{i + 1}.wav"
                output_file_path = os.path.join(output_dir, segment_filename)

                # 保存子剪辑
                segment.write_audiofile(output_file_path)

    print(f"音频分割完成，所有MP4文件的音频部分已保存到 {output_dir}")


# 示例使用
folder_path = "G:\\class_object\\learning_note\\nerual_network\\sound"  # 输入的文件夹路径，包含所有MP4文件
output_dir = "G:\\class_object\\learning_note\\nerual_network\\output_audio_segments"  # 输出的文件夹路径

split_audio_of_mp4_files_in_folder(folder_path, output_dir)
from pydub import AudioSegment
import numpy as np
import os
from pydub import AudioSegment

# # 加载 MP4 文件
# audio = AudioSegment.from_file(r"D:\File\PycharmProject\NeuroScience\sounds\6\horse.mp4", format="mp4")
#
# # 导出为 WAV 文件
# audio.export(r"D:\File\PycharmProject\NeuroScience\sounds\6\horse.wav", format="wav")
#


def detect_and_trim_audio(input_path, output_path, silence_thresh=-40, chunk_size=10):
    """
    检测音频中有声音的部分并剪辑出来。

    参数：
    - input_path: 输入音频文件路径
    - output_path: 输出音频文件路径
    - silence_thresh: 判定是否有声音的音量阈值（单位为 dB，负值越大越灵敏）
    - chunk_size: 处理音频的块大小（单位为毫秒）

    返回：
    - trimmed_audio: 剪辑后的音频片段
    """
    # 加载音频
    audio = AudioSegment.from_file(input_path)

    # 初始化有声音的片段
    non_silent_chunks = []

    # 将音频分成若干块，逐块检测
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i : i + chunk_size]

        # 计算块的平均音量
        if chunk.dBFS > silence_thresh:
            non_silent_chunks.append(chunk)

    # 将所有有声音的片段拼接起来
    if non_silent_chunks:
        trimmed_audio = sum(non_silent_chunks)

        # 导出剪辑后的音频
        trimmed_audio.export(output_path, format="wav")
        print(f"Trimmed audio saved to {output_path}")
    else:
        print("No audio detected above silence threshold.")
        trimmed_audio = None

    return trimmed_audio


input_folder_path = r"D:\File\PycharmProject\NeuroScience\sounds\6"
output_folder_path = r"D:\File\PycharmProject\NeuroScience\sounds\5"
for filename in os.listdir(input_folder_path):
    input_file_path = os.path.join(input_folder_path, filename)
    output_file_path = os.path.join(output_folder_path, filename)
    if os.path.isfile(input_file_path):
        detect_and_trim_audio(
            input_file_path, output_file_path, silence_thresh=-40, chunk_size=10
        )

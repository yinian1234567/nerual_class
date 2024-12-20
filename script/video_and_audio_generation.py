import os
import numpy as np
import random

from imageio.config.extensions import video_extensions
from jedi.inference.finder import filter_name
from openpyxl.styles.builtins import total
from pydub import AudioSegment
from moviepy.editor import ImageSequenceClip
from pyglet import image

# 最后视频或音频文件名格式：video_{n}_{ratio}

# vision_major_path = r"D:\File\PycharmProject\NeuroScience\images\cat"
# vision_other_path = r"D:\File\PycharmProject\NeuroScience\images\notcat"
#
# sound_major_path = r"D:\File\PycharmProject\NeuroScience\sounds\cat"
# sound_other_path = r"D:\File\PycharmProject\NeuroScience\sounds\notcat"

vision_path = r"D:\File\PycharmProject\NeuroScience\images"
sound_path = r"D:\File\PycharmProject\NeuroScience\sounds"


# l是数组的长度，n是类别数，ratio是每个类别占比
def generate_array(l, n, ratio):
    assert len(ratio) == n + 1, "The length of ratio must be equal to n"
    assert abs(sum(ratio) - 1) < 1e-6, "The sum of ratio must be 1"
    counts = [int(l * r) for r in ratio]
    remaining = l - sum(counts)
    counts[-1] += remaining
    result = []
    for i in range(n + 1):
        result.extend([i] * counts[i])
    np.random.shuffle(result)
    return result


# arr = generate_array(10, 2, (0.4, 0.4, 0.2))


# 生成听觉函数，传入参数是：音频地址、每段音频时间（ms）、随机数组
def generate_sound(sound_path, t, arr, n, ratio):
    save_path = r"D:\File\PycharmProject\NeuroScience\sound_stim"
    combined_audio = AudioSegment.empty()
    for i in range(len(arr)):
        # 相应音频文件夹
        audio_folder = os.path.join(sound_path, str(arr[i]))
        wav_files = [f for f in os.listdir(audio_folder) if f.endswith(".wav")]
        total_time = t
        while total_time > 0:
            random_wav = random.choice(wav_files)
            random_wav_path = os.path.join(audio_folder, random_wav)
            audio = AudioSegment.from_wav(random_wav_path)
            duration_ms = len(audio)
            if duration_ms > total_time:
                combined_audio += audio[:total_time]
                total_time = 0
            else:
                combined_audio += audio
                total_time -= duration_ms
    file_name = f"sound_{n}" + "_" + "_".join(f'{r}' for r in ratio) + ".wav"
    combined_audio.export(os.path.join(save_path, file_name), format="wav")
    return

# generate_sound(sound_path, 100, generate_array(10, 2, (0.4, 0.4, 0.2)), 2, (0.4, 0.4, 0.2))


def generate_vision(vision_path, fps, arr, n, ratio):
    save_path=r"D:\File\PycharmProject\NeuroScience\vision_stim"
    selected_images=[]
    for i in range(len(arr)):
        image_folder = os.path.join(vision_path, str(arr[i]))
        image_files=[ f for f in os.listdir(image_folder) if f.endswith(".png") ]
        random_img=random.choice(image_files)
        random_img_path = os.path.join(image_folder, random_img)
        selected_images.append(random_img_path)
    file_name=f"vision_{n}"+"_"+"_".join(f'{r}' for r in ratio)+".mp4"
    video_clip = ImageSequenceClip(selected_images, fps=fps)
    video_clip.write_videofile(os.path.join(save_path, file_name),codec='libx264')
    return

t=10
fps=2
n=2
ratio=(0.4,0.4,0.2)
arr=generate_array(20,n,ratio)
generate_sound(sound_path,1000/fps,arr,n,ratio)
generate_vision(vision_path,fps,arr,n,ratio)

# generate_vision(vision_path,20,generate_array(10,2,(0.4,0.4,0.2)),2,(0.4,0.4,0.2))
# # 生成听觉总刺激，传入的参数为总时长t(单位是s)和主物体声音出现的比例ratio
# def generate_hearing(major_stim_folder_path, other_stim_folder_path, t, ratio):
#     # 初始化一个空音频对象，用于存储合并后的音频
#     combined_audio = AudioSegment.empty()
#     major_stim_ms, other_stim_ms = 1000 * t * ratio, 1000 * t * (1 - ratio)
#     major_wav_files = [
#         f for f in os.listdir(major_stim_folder_path) if f.endswith(".wav")
#     ]
#     other_wav_files = [
#         f for f in os.listdir(other_stim_folder_path) if f.endswith(".wav")
#     ]
#     major_wavs, other_wavs = [], []
#     major_t, other_t = 0, 0
#     while major_t < major_stim_ms:
#         # 随机选一个.wav文件
#         random_wav = random.choice(major_wav_files)
#         # 获得这个.wav文件的完整路径
#         random_wav_path = os.path.join(major_stim_folder_path, random_wav)
#         major_wavs.append(random_wav_path)
#         audio = AudioSegment.from_file(random_wav_path)
#         major_t += len(audio)
#     while other_t < other_stim_ms:
#         other_wav = random.choice(other_wav_files)
#         other_wav_path = os.path.join(other_stim_folder_path, other_wav)
#         other_wavs.append(other_wav_path)
#         audio = AudioSegment.from_file(random_wav_path)
#         other_t += len(audio)
#     major_wavs, other_wavs = np.array(major_wavs), np.array(other_wavs)
#     wavs = np.concatenate((major_wavs, other_wavs))
#     np.random.shuffle(wavs)
#     for wav in wavs:
#         audio = AudioSegment.from_file(wav)
#         combined_audio += audio
#     combined_audio.export(
#         rf"D:\File\PycharmProject\NeuroScience\sound_stim\sound_{ratio}.wav",
#         format="wav",
#     )
#     return combined_audio

#
# # 生成视觉总刺激的视频，传入参数是主要刺激和其他刺激的文件夹地址，持续时长（单位s），比例ratio和每秒帧率fps
# def generate_vision(major_stim_folder_path, other_stim_folder_path, t, ratio, fps):
#     frame_duration = 1.0 / float(fps)
#     n = t * fps
#     major_num, other_num = int(n * ratio), int(n - n * ratio)
#     major_images = [
#         os.path.join(major_stim_folder_path, img)
#         for img in os.listdir(major_stim_folder_path)
#     ]
#     major_images = major_images * (major_num // len(major_images) + 1)
#     selected_major_images = random.sample(major_images, major_num)
#     other_images = [
#         os.path.join(other_stim_folder_path, img)
#         for img in os.listdir(other_stim_folder_path)
#     ]
#     other_images = other_images * (other_num // len(other_images) + 1)
#     selected_other_images = random.sample(other_images, other_num)
#     selected_images = selected_major_images + selected_other_images
#     random.shuffle(selected_images)
#     video_clip = ImageSequenceClip(selected_images, fps=fps)
#     video_clip.write_videofile(
#         rf"D:\File\PycharmProject\NeuroScience\vision_stim\vision_{ratio}.mp4",
#         codec="libx264",
#     )
#     return video_clip

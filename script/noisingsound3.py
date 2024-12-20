import random
from pydub import AudioSegment
from pydub.effects import compress_dynamic_range
import os
import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln
from scipy import signal

# 数据结构对齐：
# 单声道和立体声的音频数据结构是不同的。单声道音频数据是一个一维数组，而立体声音频数据是一个二维数组，每个通道的数据分别存储在不同的行中。
# 如果通道数不一致，直接将这些音频数据混合会导致数据结构不匹配，无法正确执行数学运算。
# 播放一致性：
# 播放设备通常期望接收具有固定通道数的音频数据。如果混合后的音频文件通道数不一致，播放设备可能无法正确解析和播放音频。


# 声音加噪模糊：
# 1. 设计并应用带通滤波器
# 作用：
# 滤波的目的是在频率域内选取一个特定的频率范围（low_freq ~ high_freq），并移除信号中的其他频率成分。
# 带通滤波器只允许位于特定频率范围内的信号通过，而衰减其他频率的信号。这样可以去掉语音中特定的高频噪音或低频成分，保留一个有限的频率带宽，从而模糊语音特征。
# 2. 降低采样率（重采样）
# 作用：
# 通过降低采样率减少音频的细节信息，使得声音质量下降，进一步模糊语音特征。
# 降低采样率会导致可表达的最高频率（奈奎斯特频率）下降，也就是丢失部分高频信息。


# 1. 滤波器的频率归一化
# 滤波器上下界的归一化是相对于奈奎斯特频率（0.5 * sample_rate）进行的，目的是将频率范围映射到 [0, 1] 区间，以满足滤波器设计的要求。
# 这种归一化只影响滤波器的参数计算，与音频信号的值无关

# 如果某些音频的尾部含有较大的振幅值，那么这种截断会导致以下问题：
#  原始音频的尾部可能包含突出的声音片段，截断后这些信息丢失，表现为幅值的变化。
#  在实际音频处理中，将所有音频数据长度对齐到最短的音频长度可能不适用于某些场景。因为不同音频的内容结构可能不同，截断音频可能会造成：
#     丢失重要信息（例如尾部音量渐变）。
#     改变音频整体能量分布。
# 波形不完整
# 如果截断时音频的周期性信号（例如，正弦波的一个周期）被打断，可能导致波形在视觉和听觉上发生变化。这种变化可能被误认为是幅值的改变

# 音频刺激的量化：
# 归一化处理使得分贝的影响可以忽略
# 因为音频的变化是通过添加噪声进行的，所以使用信噪比来量化证据量,使用当noise_level=0时的函数的输出音频，作为标准的原始音频用于计算信噪比


def ensure_audio_length(audio, target_length):
    current_length = len(audio)
    if current_length < target_length:
        repeat_times = (target_length + current_length - 1) // current_length
        audio = np.tile(audio, repeat_times)[:target_length]
    else:
        audio = audio[:target_length]
    return audio


def match_audio_channels(audio, target_channels):
    if target_channels == 1 and audio.ndim == 2:
        return np.mean(audio, axis=1)  # 转为单声道
    elif target_channels == 2 and audio.ndim == 1:
        return np.column_stack((audio, audio))  # 单声道转为立体声
    return audio  # 已经匹配


def match_audio_length(audio_list,targetlen=0):
    # 获取所有音频的长度
    lengths = [len(audio) for audio in audio_list]

    # 计算所有音频长度的最小公倍数
    target_length = len( audio_list[0])
    # 重复填充音频到目标长度
    padded_audio_list = []
    for audio, length in zip(audio_list, lengths):
        repeat_count = -(-target_length // length)  # 向上取整计算重复次数
        padded_audio = np.tile(audio, repeat_count)[:target_length]  # 截断到目标长度
        padded_audio_list.append(padded_audio)

    return padded_audio_list


def normalize_audio(audio, target_range=(-1.0, 1.0)):
    # 确保音频数据是浮点型，避免整数溢出
    audio = audio.astype(np.float32)

    # 找到音频的最大绝对值
    max_val = np.max(np.abs(audio))

    # 如果音频已经是静音（全为零），直接返回
    if max_val == 0:
        return audio

    # 将音频数据归一化到 [-1, 1] 范围
    audio_normalized = audio / max_val

    # 根据目标范围进行缩放和偏移
    max_val = np.max(audio_normalized)
    min_val = np.min(audio_normalized)
    min_target, max_target = target_range
    if max_val - min_val != 0:
        audio_normalized = (audio_normalized - (max_val + min_val) / 2) * (max_target - min_target) / (
                    max_val - min_val)

    return audio_normalized


def calculate_snr(standard_sound_path, noisy_signal):
    if standard_sound_path == '':
        return 0
    # 确保两段信号长度一致
    signal, sample_rate = sf.read(standard_sound_path)

    # 计算噪声
    noise = noisy_signal - signal

    # 计算信号功率和噪声功率
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)

    # 计算 SNR
    snr = 10 * np.log10(signal_power / noise_power)

    return snr


## 打乱区域内的值进行信息增量正负变化
def shuffle_in_intervals(z, num_intervals=5):
    # 获取 z 的长度
    n = len(z)

    # 计算每个区间的长度
    interval_length = n // num_intervals

    # 初始化一个空列表来存储打乱后的大区间
    shuffled_z = []

    # 遍历 z，每次取 interval_length 个元素作为一个大区间
    for i in range(0, n, interval_length):
        # 获取当前大区间
        interval = z[i:i + interval_length]

        # 打乱当前大区间的顺序
        random.shuffle(interval)

        # 将打乱后的当前大区间添加到结果列表中
        shuffled_z.extend(interval)

    # 处理剩余的元素（如果有的话）
    remaining_elements = n % num_intervals
    if remaining_elements > 0:
        interval = z[-remaining_elements:]
        random.shuffle(interval)
        shuffled_z.extend(interval)

    return shuffled_z


def measure_max_loudness(y):
    """计算给定音频数据y的最大响度（单位：dB）"""
    max_amplitude = np.max(np.abs(y))
    return librosa.amplitude_to_db(max_amplitude, ref=np.max)


def adjust_loudness_to_target(y, target_loudness):
    """调整音频y使其达到目标最大响度target_loudness（单位：dB）"""
    current_loudness = measure_max_loudness(y)
    gain = 10 ** ((target_loudness - current_loudness) / 20.0)
    adjusted_y = y * gain
    # 防止削波
    if np.max(np.abs(adjusted_y)) > 1.0:
        adjusted_y = np.clip(adjusted_y, -1.0, 1.0)
    return adjusted_y


def apply_dynamic_range_compression(y, sr):
    """应用动态范围压缩"""
    # 将numpy数组转换为AudioSegment对象
    audio = AudioSegment(y.tobytes(), frame_rate=sr, sample_width=y.dtype.itemsize, channels=1)

    # 应用动态范围压缩
    compressed_audio = compress_dynamic_range(audio, threshold=-20.0, ratio=4.0, attack=5.0, release=50.0)

    # 将AudioSegment对象转换回numpy数组
    compressed_y = np.array(compressed_audio.get_array_of_samples()).astype(np.float32) / (
                2 ** (8 * y.dtype.itemsize - 1))

    return compressed_y


# 响度调整函数
def normalize_loudness(audio, sample_rate, target_loudness=-23.0):
    """
    将音频的响度调整到目标响度。

    :param audio: 输入音频数据
    :param sample_rate: 音频采样率
    :param target_loudness: 目标响度 (LUFS)
    :return: 调整后的音频数据
    """
    meter = pyln.Meter(sample_rate)  # 使用采样率初始化响度测量工具
    current_loudness = meter.integrated_loudness(audio)  # 计算当前响度
    loudness_offset = target_loudness - current_loudness  # 计算所需增益
    adjusted_audio = pyln.normalize.loudness(audio, current_loudness, target_loudness)
    current_loudness = meter.integrated_loudness(adjusted_audio)  # 计算当前响度
    return adjusted_audio# 响度调整函数


def blur_audio(input_file, secondary_audio_paths, output_file='', noise_level=0.01, low_freq=300, high_freq=1500,
               duration=5, standard_sound_path=''):
    # 读取主音频文件
    # 主处理逻辑
    audio, sample_rate = sf.read(input_file,dtype='float32')  # 读取主音频文件

    # 调整主音频的响度
    audio = normalize_loudness(audio, sample_rate)
    targetlen=len(audio)

    secondary_audio_data_list = []
    output_dir = './output'  # 定义输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 如果输出目录不存在，则创建

    for i, path in enumerate(secondary_audio_paths):
        # 读取次要音频文件
        secondary_audio_data,secondary_sample_rate = sf.read(path,dtype='float32')

        # 如果采样率不匹配，则重采样
        if sample_rate != secondary_sample_rate:
            secondary_audio_data = librosa.resample(secondary_audio_data.astype(np.float32),
                                                    orig_sr=secondary_sample_rate,
                                                    target_sr=sample_rate)

        # 调整声道数量
        secondary_audio_data = match_audio_channels(secondary_audio_data, target_channels=1)

        # 调整响度到目标值
        secondary_audio_data = normalize_loudness(secondary_audio_data, sample_rate)

        # 添加到列表
        # secondary_audio_data = normalize_audio(secondary_audio_data)
        secondary_audio_data_list.append(secondary_audio_data)


    audio = match_audio_channels(audio, target_channels=1)

    # audio = normalize_audio(audio)

    # 对齐长度
    all_audio_data = [audio] + secondary_audio_data_list
    aligned_audio = match_audio_length(all_audio_data,targetlen=targetlen)
    audio, *secondary_audio_data_list = aligned_audio

    # 如果是多通道音频，只处理第一通道
    if len(audio.shape) > 1:
        audio = audio[:, 0]

    # 加入高斯噪声,noise_level 控制了噪声的标准差，进而影响噪声的强度
    noise = np.zeros_like(audio, dtype=np.float32)
    # 这里的音频都进行了归一化，使得分贝大小类似
    for secondary_audio in secondary_audio_data_list:
        noise += secondary_audio
    noise -= audio
    # 添加高斯噪声
    noise += np.random.normal(scale=noise_level*noise_level, size=noise.shape)
    noise *= noise_level  # 调整噪声比例
    # 调整噪声的响度
    audio_noisy = audio + noise

    # 设计一个带通滤波器 (low_freq ~ high_freq)
    nyquist = 0.5 * sample_rate
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = signal.butter(5, [low, high], btype='band')
    audio_filtered = signal.filtfilt(b, a, audio_noisy)

    # 降低采样率 (重采样)
    new_sample_rate = sample_rate // 2
    audio_resampled = signal.resample(audio_filtered, int(len(audio_filtered) * new_sample_rate / sample_rate))
    audio_resampled = normalize_audio(audio_resampled)
    audio_resampled = normalize_audio(audio_noisy)

    # 调整音频长度
    target_length = int(duration * sample_rate)  # 目标长度，单位：采样点数
    # target_length = int(duration * new_sample_rate)  # 目标长度，单位：采样点数
    current_length = len(audio_resampled)

    if current_length >= target_length:
        # 如果当前长度大于目标长度，裁剪至目标长度
        final_audio = audio_resampled[:target_length]
    else:
        # 如果当前长度不足目标长度，循环填充
        repeat_times = (target_length + current_length - 1) // current_length
        final_audio = np.tile(audio_resampled, repeat_times)[:target_length]


    # audio是经过归一化后的原始信号，这里使用final作为加噪后的信号
    snr = calculate_snr(standard_sound_path, final_audio)
    # return final_audio, new_sample_rate, snr
    return final_audio, sample_rate, snr


def getlongaudio(input_file, out_file, fps=0.5):
    # 每个音频所需的时间
    single_audio_time = fps ** (-1)
    # 创建保存文件夹
    output_dir = out_file
    # 素材文件夹
    factor_path = input_file
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    label = ['cat', 'dog', 'fox','horse','deer']
    dict = {'cat': '0', 'dog': '1', 'fox': '2','deer':"3",'horse':"4"}
    # 这里都用第一个文件，方便
    for i in label:
        main_audio_path = os.path.join(factor_path, dict[i], i + '_0.wav')
        secondary_audio_paths = []
        for j in label:
            if i == j:
                continue
            second_audio_path = os.path.join(factor_path, dict[j], j + '_0.wav')
            secondary_audio_paths.append(second_audio_path)

        # 生成standard文件用于后续的snr计算：
        standard_sound_path = os.path.join(output_dir, i, 'standard_' + 'label_' + i + '.wav')
        if not os.path.isdir(os.path.join(output_dir, i)):
            os.makedirs(os.path.join(output_dir, i))

        final_audio, new_sample_rate, snr = blur_audio(main_audio_path, secondary_audio_paths,
                                                       duration=single_audio_time,
                                                       output_file=standard_sound_path, noise_level=0)
        sf.write(standard_sound_path, final_audio, new_sample_rate)
        print(f"模糊处理完成，结果保存至 {standard_sound_path}")

        # 初始化变量
        snr_list = []
        noise_level_list = []
        total_audio = None

        for z in shuffle_in_intervals(np.arange(0.35, -0.05, -0.05), num_intervals=5):
            # 生成单个音频及计算SNR
            final_audio, new_sample_rate, snr = blur_audio(main_audio_path, secondary_audio_paths,
                                                           duration=single_audio_time,
                                                           noise_level=z, standard_sound_path=standard_sound_path)

            # 确保final_audio的数据类型为int16
            final_audio = (final_audio * 32767).astype(np.int16)

            # 转换numpy数组到pydub的AudioSegment对象
            audio_segment = AudioSegment(final_audio.tobytes(),
                                         frame_rate=new_sample_rate,
                                         sample_width=final_audio.dtype.itemsize,
                                         channels=1 if len(final_audio.shape) == 1 else final_audio.shape[1])

            # 使用淡入淡出技术平滑过渡
            fade_duration = 100  # 淡入淡出持续时间，单位毫秒
            audio_segment = audio_segment.fade_in(fade_duration).fade_out(fade_duration)

            # 如果这是第一个音频，则直接赋值给total_audio
            if total_audio is None:
                total_audio = audio_segment
            else:
                # 否则，将新音频追加到total_audio
                total_audio += audio_segment

            # 记录当前音频的SNR
            snr_list.append(snr)
            noise_level_list.append(z)

        # 将所有音频合并后的结果导出
        if total_audio is not None:
            combined_output_path = os.path.join(output_dir, i, f'combined_audio.wav')
            total_audio.export(combined_output_path, format="wav")
            print(f"所有音频合并完成，结果保存至 {combined_output_path}")

        # 将SNR列表和噪声级别写入文本文件
        snr_file_path = os.path.join(output_dir, i, f'snr_values.txt')
        with open(snr_file_path, 'w') as f:
            for snr, noise_level in zip(snr_list, noise_level_list):
                f.write(f"SNR: {snr}, Noise Level: {noise_level}\n")

        print(f"SNR值和噪声级别已保存至 {snr_file_path}")


output_dir = r"G:\NeuroScience (2)\sound_stim"
# 素材文件夹
factor_path = r"G:\NeuroScience (2)\newsounds"
getlongaudio(factor_path, output_dir, fps=0.5)

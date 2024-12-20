import random
from pydub import AudioSegment
from pydub.effects import compress_dynamic_range
import os
import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln
import re
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


def calculate_snr(signal, noisy_signal):
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


def blur_audio(input_file, secondary_audio_paths, noise_level=0.01, duration=5):
    # 读取主音频文件
    # 主处理逻辑
    audio, sample_rate = sf.read(input_file,dtype='float32')  # 读取主音频文件

    # 调整主音频的响度
    audio = normalize_loudness(audio, sample_rate)
    targetlen=len(audio)

    secondary_audio_data_list = []
    output_dir = '../output'  # 定义输出目录
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
    noise *= noise_level  # 调整噪声比例
    # 调整噪声的响度
    audio_noisy = audio + noise

    audio_resampled = normalize_audio(audio_noisy)
    # 调整音频长度
    target_length = int(duration * sample_rate)  # 目标长度，单位：采样点数
    current_length = len(audio_resampled)

    if current_length >= target_length:
        # 如果当前长度大于目标长度，裁剪至目标长度
        final_audio = audio_resampled[:target_length]
    else:
        # 如果当前长度不足目标长度，循环填充
        repeat_times = (target_length + current_length - 1) // current_length
        final_audio = np.tile(audio_resampled, repeat_times)[:target_length]


    return final_audio, sample_rate

def get_choice_audio(input_file, secondary_audio_paths, noise_level=0.01, duration=5):
    # 读取主音频文件
    # 主处理逻辑
    audio, sample_rate = sf.read(input_file,dtype='float32')  # 读取主音频文件

    # 调整主音频的响度
    audio = normalize_loudness(audio, sample_rate)
    targetlen=len(audio)

    secondary_audio_data_list = []
    output_dir = '../output'  # 定义输出目录
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
    noise *= noise_level  # 调整噪声比例
# change place
    # 调整噪声的响度
    audio_noisy = noise

    audio_resampled = normalize_audio(audio_noisy)
    # 调整音频长度
    target_length = int(duration * sample_rate)  # 目标长度，单位：采样点数
    current_length = len(audio_resampled)

    if current_length >= target_length:
        # 如果当前长度大于目标长度，裁剪至目标长度
        final_audio = audio_resampled[:target_length]
    else:
        # 如果当前长度不足目标长度，循环填充
        repeat_times = (target_length + current_length - 1) // current_length
        final_audio = np.tile(audio_resampled, repeat_times)[:target_length]
    return final_audio, sample_rate

# 使用blur_audio计算结果音频，然后以get_choice_audio计算每个时间步的选项音频，其中secondary_audio_paths只含有选项包含的类
def getlongaudio(input_file, noise_levels,choices,truechoice,laststr="_0.wav",fps=0.5):
    single_audio_time = fps ** (-1)
    # 素材文件夹
    factor_path = input_file
    label = ['cat', 'dog', 'fox','horse','deer']
    dict = {'cat': '0', 'dog': '1', 'fox': '2','deer':"3",'horse':"4"}
    choice0_paths=os.path.join(factor_path,dict[choices[0]], choices[0] + laststr)
    choice1_paths=os.path.join(factor_path,dict[choices[1]], choices[1] + laststr)
    choice_paths={0:[choice0_paths],1:[choice1_paths]}
    # 这里都用第一个文件，方便
    main_audio_path = os.path.join(factor_path, dict[truechoice], truechoice + laststr)
    choicenone_paths=[]
    secondary_audio_paths = []
    for j in label:
        if truechoice== j:
            continue
        second_audio_path = os.path.join(factor_path, dict[j], j + laststr)
        secondary_audio_paths.append(second_audio_path)
        if j in choices:
            continue
        choicenone_path = os.path.join(factor_path, dict[j], j + laststr)
        choicenone_paths.append(choicenone_path)
    # 初始化变量
    snr_choice_list = {0:[],1:[]}
    snrnone_list = []
    for z in noise_levels:
        # 生成混合音频
        mix_audio, new_sample_rate= blur_audio(main_audio_path, secondary_audio_paths,
                         duration=single_audio_time, noise_level=z)
        # 生成选项音频
        for index,i in enumerate(choices):
            # 如果选项为正确答案音频，那么不求他的snr，因为结果文件有了
            if i==truechoice:
                standard_audio, news_sample_rate = blur_audio(main_audio_path, secondary_audio_paths,
                        duration=single_audio_time, noise_level=0)
                snr = calculate_snr( standard_audio,mix_audio)
                snr_choice_list[index].append(snr)
                continue
            choice_audio, newc_sample_rate = get_choice_audio(main_audio_path, choice_paths[index],
                         duration=single_audio_time,noise_level=z)
            snr=calculate_snr(choice_audio,mix_audio)
            snr_choice_list[index].append(snr)
        # 生成none选项的
        choice_audio, newc_sample_rate = get_choice_audio(main_audio_path, choicenone_paths,
                                                          duration=single_audio_time,
                                                          noise_level=z)
        snr = calculate_snr(choice_audio,mix_audio)
        snrnone_list.append(snr)
    return snr_choice_list,snrnone_list




def extract_noise_levels(file_path):
    # 创建一个空列表来存储噪声水平
    noise_levels = []

    # 定义正则表达式模式来匹配 Noise Level 后面的数字
    pattern = r'Noise Level: ([\d.]+)'

    # 打开文件并读取内容
    with open(file_path, 'r') as file:
        content = file.read()

        # 查找所有匹配项
        matches = re.findall(pattern, content)

        # 遍历所有匹配项，转换为浮点数并保留两位小数
        for match in matches:
            noise_level = float(match)
            noise_levels.append(round(noise_level, 2))

    return noise_levels


# 目前模拟的就是1素材的cat,dog实验

# 先使用试验结果获得对应的noise_level顺序，不用snr数据，后续会重新计算
# changeplace：需要改存储实验结果的文件路径
dirname={1:"fox",2:"cat",3:"dog",4:"deer",5:"horse"}
file_path = r'/modeling/第二部分实验/1/snr_values.txt'
noise_levels = extract_noise_levels(file_path)

# 确定实验的选项与正确选项，因为正确选项的音视之间需要进行加权处理
# changeplace
choices=["dog","cat"]
truechoice="cat"

# 使用素材文件夹取计算每个选项的snr,其中laststr="_0.wav",决定我们使用的是1,2,3,素材文件夹的哪一个，1对应参数的0...
# 返回值获取了对应的每个选项的snr，这里由于我们在多因素的ddm里面，不认为其他声音对选择有影响，所以不管这里的snrnone_list
# changeplace:这里只需要改laststr="_0.wav"
factor_path = r"/newsounds"
snr_choice_list,snrnone_list=getlongaudio(factor_path, noise_levels,choices,truechoice,laststr="_1.wav",fps=0.5)
print(f"{snr_choice_list[0]}\n{snr_choice_list[1]}\n")


# 保存到文件
lists_save = [
    (f"{choices[0]}",snr_choice_list[0]),
    (f"{choices[1]}",snr_choice_list[1]),
]

# 听觉提供的drift文件保存在"cat_dog_drift"
# changeplace：改存储路径,保存的文件名表示正确选项，利于后续使用
save_path=os.path.join(os.path.dirname(__file__), "cat_dog_drift")
if not os.path.isdir(save_path):
    os.mkdir(save_path)
save_file=os.path.join(save_path,truechoice+".txt")
# 打开文件准备写入
with open(save_file, 'w') as file:
    for identifier, lst in lists_save:
        # 构造字符串并写入文件
        formatted_string = f"{identifier}:{','.join(map(str, lst))}"
        file.write(formatted_string + '\n')

print("Lists saved to output.txt")




import numpy as np
import os

def read_and_sort_numbers(file_path,step_time=0.1):
    """
    要求输入的step_time是0.5的因子
    """
    # 用于存储所有数值的列表
    numbers = []

    # 打开文件并逐行读取
    with open(file_path, 'r') as file:
        for line in file:
            # 使用空格分割每行中的数值字符串，并转换为浮点数
            nums = [float(num) for num in line.strip().split()]
            # 将当前行的所有数值添加到总列表中
            numbers.extend(nums)

    # 使用numpy数组进行排序，然后反转以获得降序
    sorted_numbers = np.sort(numbers)
    scale=int(0.5/step_time)
    attention=np.zeros(len(sorted_numbers)*scale)
    for i in range(0,len(sorted_numbers)):
        for j in range(0,scale):
            attention[scale*i+j]=sorted_numbers[i]


    return attention

def change_snr_dict(snr_dict,step_time=0.1):
    """
    要求输入的step_time是2的因子
    """
    num=len(snr_dict)
    keys_drift_dict = list(drift_dict.keys())
    scale=int(2/step_time)

    # 创建扩展为时间步长度的字典
    items = list(zip(keys_drift_dict ,np.zeros((num,len(snr_dict[keys_drift_dict[0]])*scale))))
    snr = dict(items)

    for i in range(0,num):
        for j in range(0, len(snr_dict[keys_drift_dict[0]])):
            for z in range(0,scale):
                snr[keys_drift_dict[i]][scale*j+z]=snr_dict[keys_drift_dict[i]][j]
    return snr

def getdict(path):
    # 初始化一个空字典用于存储结果
    data_dict = {}

    # 使用with语句打开文件，这样可以自动处理文件关闭
    with open(path, 'r') as file:
        # 逐行读取文件内容
        for line in file:
            # 去除可能存在的换行符，并按冒号分割行内容
            identifier, data_str = line.strip().split(':')

            # 将数据字符串分割成列表，并转换为浮点
            data_list = list(map(float, data_str.split(',')))

            # 将标识符和数据列表添加到字典中
            data_dict[identifier] = data_list

    # 输出最终的字典以验证
    return data_dict


# 存储视觉权重的地方，即实验结果路径
# changeplace：
main_path=os.path.join(os.path.dirname(__file__), "get_multchoice_result")
dirname={1:"fox",2:"cat",3:"dog",4:"deer",5:"horse"}
f_dirname={"fox":1,"cat":2,"dog":3,"deer":4,"horse":5}
given_drift_rates=[]
# 指定保存路径和文件名

weight=[10,1]
bias=[1,10]


# 获取1,2,3,4,5，实验的结果的漂移率，将其snr与attention加权组合后转化为时间步数组后保存
# # 存储听觉的地方
# changeplace
factor_dir=os.path.join(os.path.dirname(__file__), "dog_fox_drift")
# 文件的形式是：
"""
'dog': [1, 2, 3, 4],
'fox': [5, 6, 7, 8],
'none': [9, 10, 11, 12]
"""

# 保存得到的最终的drift的路径
save_path = os.path.join(os.path.dirname(__file__), "dog_fox_drift_timelen")
if not os.path.isdir(save_path):
    os.mkdir(save_path)
# 下面这个循环遍历每个实验，得到对应的drift并保存，且得到每个

for i in os.listdir(factor_dir):
    # 获取视觉的drift
    base_name, extension = os.path.splitext(i)
    att_path=os.path.join(main_path,str(f_dirname[base_name]),"weight_matrix.txt")
    att_array = read_and_sort_numbers(att_path)

    # 获取听觉的drift
    path=os.path.join(factor_dir,i)
    drift_dict=getdict(path)
    keys_drift_dict= list(drift_dict.keys())
    drift_dict_snr=change_snr_dict(drift_dict)

    # 计算得到时间步为长度的含有各个选项drift的字典
    min_len=min(len(att_array),len(drift_dict_snr[base_name]))
    att_array=att_array[:min_len]
    for j in keys_drift_dict:
        drift_dict_snr[j]=(drift_dict_snr[j][:min_len]+bias[1])*weight[1]
    drift_dict_snr[base_name]+=(att_array+bias[0])*weight[0]

    save_file = os.path.join(save_path,i)
    # 打开文件，准备写入
    with open(save_file, 'w') as file:
        # 遍历字典中的每一个键值对
        for key, value in drift_dict_snr.items():
            # 将列表转换为字符串，并且在每个元素之间加入空格
            value_str = ','.join(map(str, value))
            # 写入文件，格式为 "key: value"，其中value是列表的字符串表示
            file.write(f"{key}: {value_str}\n")







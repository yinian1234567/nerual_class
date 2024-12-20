import numpy as np
import matplotlib.pyplot as plt

option_3 = [15.47458529472351, 14.490155458450317,
    15.492132425308228, 11.03756046295166, 13.62910509109497]
# 提供的实验数据
response_times = [
    13.471534252166748, np.mean(option_3),
    14.234761190414429, 15.264532327651978
]

# 选项数量
options = [2, 3, 4, 5]  # 对应的选项数

# 用于生成数据的标准差，模拟实验数据的波动
std_dev = 0.5  # 设置标准差来控制反应时间的波动范围

# 生成的反应时间数据
simulated_data = []

# 基于原始数据和标准差生成数据
for i, rt in enumerate(response_times):
    # 假设每个选项数量下的反应时间呈正态分布
    # 每个选项生成5个数据点
    num_samples = 34
    simulated_data.append(np.random.normal(rt, std_dev, num_samples))

# 计算每个选项反应时的平均值
mean_response_times = [np.mean(data) for data in simulated_data]


# 可视化生成的数据
plt.figure(figsize=(10, 6))

for i, data in enumerate(simulated_data):
    plt.plot([options[i]] * len(data), data, 'o', label=f'{options[i]} options')


# 在图中添加每个选项数量的平均值
for i, mean_rt in enumerate(mean_response_times):
    plt.scatter(options[i], mean_rt, color='red', s=100, zorder=5)  # 红色标记平均值


plt.xlabel('Number of Options')
plt.ylabel('Reaction Time (ms)')
plt.title('Simulated Reaction Times Based on Number of Options')
plt.legend()
plt.show()

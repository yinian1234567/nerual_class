import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from get_drift_for_multchoice_foxanddog import getdict


def get_threshold(given_drift_rates, time_steps, dt,weight,choice_name,rt,sigma=50):
    """

    :param given_drift_rates:列表，(num_choices,时间步长度)
    :param time_steps:
    :param dt:
    :param threshold:列表，(numchoices,)
    :param o:
    :weight:是减去其他选项的信息增益的权重矩阵 ：wij,表是j选项的信息增量对i选项的信息增量的影响
    :return:
    """

    num_choices=len(choice_name)
    # 创建一系列独立的零数组
    arrays = [np.zeros(time_steps) for _ in range(len(choice_name))]
    items = list(zip(choice_name, arrays))
    decision_paths =dict(items)  # 决策路径，形状为 dict

    decision_times = 0.0
    decision=""
    threshold=0.0
    for t in range(1, time_steps):
        if (t * dt <= rt):
            for i in choice_name:
                decision_paths[i][t] += weight[i][i]*given_drift_rates[i][t] * dt+decision_paths[i][t-1]+sigma*dt*np.random.normal()*np.sqrt(dt)
            for i in choice_name:
                for j in choice_name:
                    if j==i:
                        continue
                    decision_paths[i][t]-=weight[i][j]*given_drift_rates[j][t]*dt+sigma*dt*np.random.normal()*np.sqrt(dt)
        elif (t*dt>=rt):
            for i in choice_name:
                decision_paths[i][t] += weight[i][i] * given_drift_rates[i][t] * (t*dt-rt) + decision_paths[i][
                    t - 1]
            for i in choice_name:
                for j in choice_name:
                    if j == i:
                        continue
                    decision_paths[i][t] -= weight[i][j] * given_drift_rates[j][
                        t] *(t*dt-rt)
            for i in choice_name:
                threshold =max(decision_paths[i][t],threshold)
            break
    return threshold


# 漂移扩散模型
def drift_diffusion_model(given_drift_rates, time_steps, dt, threshold,weight,choice_name,sigma=50):
    """

    :param given_drift_rates:列表，(num_choices,时间步长度)
    :param time_steps:
    :param dt:
    :param threshold:列表，(numchoices,)
    :param o:
    :weight:是减去其他选项的信息增益的权重矩阵 ：wij,表是j选项的信息增量对i选项的信息增量的影响
    :return:
    """

    num_choices=len(choice_name)
    # 创建一系列独立的零数组
    arrays = [np.zeros(time_steps) for _ in range(len(choice_name))]
    items = list(zip(choice_name, arrays))
    decision_paths =dict(items)  # 决策路径，形状为 dict

    decision_times = 0.0
    decision=""
    for t in range(1, time_steps):
        for i in choice_name:
            decision_paths[i][t] += weight[i][i]*given_drift_rates[i][t] * dt+decision_paths[i][t-1]+sigma*dt*np.random.normal()*np.sqrt(dt)
        for i in choice_name:
            for j in choice_name:
                if j==i:
                    continue
                decision_paths[i][t]-=weight[i][j]*given_drift_rates[j][t]*dt+sigma*dt*np.random.normal()*np.sqrt(dt)

        for i in choice_name:
            if abs(decision_paths[i][t]) >= threshold[i] :  # 达到阈值则停止
                if decision!="" and decision_paths[i][t]<decision_paths[decision][t]:
                    continue
                decision=i
                decision_times=t*dt
        # 处理都到达了阈值的情况
        if decision!="":
            for i in choice_name:
                decision_paths[i][t:]=decision_paths[i][t]
            break


    return decision_paths,decision_times,decision

# 设置参数
num_choices = 3  # 选项数量
choice_name=["dog","fox","None"]
dt = 0.1  # 时间步长key
time_steps = int(20/dt)  # 时间步数
steps=np.linspace(0, 20,time_steps )

# 这里的权重就设置为其他的影响因子为0.5
weight={
    "dog": {"dog": 1, "fox": 0.6, "None": 0.6},
    "fox": {"dog": 0.6, "fox": 1, "None": 0.6},
    "None": {"dog": 0.6, "fox": 0.6, "None": 1}
}


reaction_time=[15.47458529472351, 14.490155458450317, 16.492132425308228, 11.03756046295166,13.62910509109497]
name=["fox","cat","dog","deer","horse"]
items=list(zip(name,reaction_time))
rt=dict(items)
############################################################################################
# 获取建模的阈值
# 遍历dog_fox_drift_timelen的每个文件，其文件名表示正确选项，因为得到的反应时都是选项选对的，
# 所以用反应时与ddm建模里面的计算方式去计算对应阈值，然后把该选项：值存入字典
criterion_dict={}
factor_dir=os.path.join(os.path.dirname(__file__), "dog_fox_drift_timelen")
for name in os.listdir(factor_dir):
    path = os.path.join(factor_dir, name)
    dict1= getdict(path)
    keys_dict = list(dict1.keys())

    basename,exp=os.path.splitext(name)


    threshold = get_threshold(dict1, time_steps, dt, weight, choice_name, rt[basename])
    print("threshold", threshold)
    criterion_dict[basename]=threshold

for i in choice_name:
    criterion_dict[i]=criterion_dict["fox"]
print(criterion_dict["fox"])

############################################################################################
# 获取建模的given_drift_rates

# 建模用的实验是dog，fox，none；fox
drift_rates_path= r"dog_fox_drift_timelen/fox.txt"
given_drift_rates=getdict(drift_rates_path)# 是一个字典

###########################################################################################
# 实验
decision_paths,decision_times,decision = drift_diffusion_model(given_drift_rates, time_steps,
                                                               dt, criterion_dict,weight,choice_name)

# 绘制每个决策路径

for key,value in decision_paths.items():
    plt.plot(steps, value, label=f'Decision Path {key}')

# 添加图例
plt.legend()

# 添加标题和轴标签
plt.title('Decision Paths over Time')
plt.xlabel('Time')
plt.ylabel('Decision Path Value')

# 显示图形
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from get_drift_for_multifactor_catanddog import getdict



# 漂移扩散模型
def drift_diffusion_model(given_drift_rates, time_steps, dt, threshold,sigma=50):
    decision_paths =np.zeros(time_steps)
    decision_times = 0.0
    decision=""
    for t in range(1, time_steps):
        decision_paths[t] += given_drift_rates[t] * dt+decision_paths[t-1]+sigma*dt*np.random.normal()*np.sqrt(dt)
        if abs(decision_paths[t]) >= threshold :  # 达到阈值则停止
            decision=1 if decision_paths[t]>0 else 0
            decision_times=t*dt
            decision_paths[t:]=decision_paths[t]
            break
    return decision_paths,decision_times,decision

# 设置参数
num_choices = 2  # 选项数量
choice_name=["dog","cat"]
dt = 0.1  # 时间步长key
time_steps = int(20/dt)  # 时间步数
steps=np.linspace(0, 20,time_steps )
drift_rates_path=[]
base_path=os.path.join(os.path.dirname(__file__),"cat_dog_drift_timelen")
drift_rates_path.append(os.path.join(base_path,"cat.txt"))
drift_rates_path.append(os.path.join(base_path,"cat_onlypic.txt"))
drift_rates_path.append(os.path.join(base_path,"dog.txt"))

given_drift_rates=[]
true_choices=["cat","cat","dog"]
false_choices=["dog","dog","cat"]
explanation=["only audio","only vision","both vision and audio"]

################################################################################################
# 阈值计算
# 遍历dog_fox_drift_timelen的每个文件，其文件名表示正确选项，因为得到的反应时都是选项选对的，
# 所以用反应时与ddm建模里面的计算方式去计算对应阈值，然后把该选项：值存入字典
reaction_time=[11.590173244476318,18.459388494491577,15.304340600967407]
name=["cat","cat_onlypic","dog"]
items=list(zip(name,reaction_time))
rt=dict(items)
weight=0.6
criterions=[]
factor_dir=os.path.join(os.path.dirname(__file__), "cat_dog_drift_timelen")
for z,name in enumerate(os.listdir(factor_dir)):
    path = os.path.join(factor_dir, name)
    dict1= getdict(path)
    keys_dict = list(dict1.keys())

    criterion=0.0
    for i in range(0,len(dict1[true_choices[z]])):
        if (i+1)*dt<rt[true_choices[z]]:
            criterion+=dt*dict1[true_choices[z]][i]
        else:
            criterion += (rt[true_choices[z]]-i*dt) * dict1[true_choices[z]][i]
        for j in keys_dict:
            if true_choices[z]==j:
                continue
            if (i + 1) * dt < rt[true_choices[z]]:
                # 这里应该选用就是weight里面的值，
                criterion -= weight*dt * dict1[j][i]
            else:
                criterion -= weight*(rt[true_choices[z]] - i * dt) * dict1[j][i]
        if (i+1)*dt>rt[true_choices[z]]:
            break

    criterions.append(criterion)

criterion=sum(i for i in criterions)/len(criterions)
print(criterion)
############################################################################################
# 获取建模的given_drift_rates
# 建模用的实验是dog，cat
for j,i in enumerate(drift_rates_path):
    drift_rates=getdict(i)
    result = [a - weight*b for a, b in zip(drift_rates[true_choices[j]], drift_rates[true_choices[j]])]
    given_drift_rates.append(np.array(result))
###########################################################################################
# 实验
total_decision_paths,total_decision_times,total_decisions=[],[],[]
for i,j in enumerate(given_drift_rates):
    decision_paths,decision_times,decision = drift_diffusion_model(j,
                                                                   time_steps,dt,criterion)
    if decision==1:
        total_decisions.append(true_choices[i])
    else:
        total_decisions.append(false_choices[i])
    total_decision_times.append(decision_times)
    total_decision_paths.append(decision_paths)

# 创建图形
fig, ax = plt.subplots()
color=["blue","red","yellow"]
# 绘制每个DDM过程
for i, path in enumerate(total_decision_paths):
    time_steps = np.arange(len(path))
    # 使用相同的颜色和样式来表示所有路径
    ax.plot(time_steps, path, color=color[i], label=f'{explanation[i]}')

# 设置图表属性
ax.set_xlabel('Time Step')
ax.set_ylabel('Evidence Accumulation')
ax.set_title('DDM Process Visualization')

# 添加图例（虽然在这里每个路径都是蓝色，但保留图例以保持一致性）
ax.legend()

# 显示图表
plt.show()
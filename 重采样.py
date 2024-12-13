import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# 读取两个TXT文件
data1 = pd.read_csv(r'C:\Users\陈宇飞\Desktop\华为杯\声音信号.txt', header=None)  # 替换为第一个TXT文件路径
data2 = pd.read_csv(r'C:\Users\陈宇飞\Desktop\华为杯\振动信号.txt', header=None)  # 替换为第二个TXT文件路径

# 创建原始索引
x_original = np.linspace(0, 1, len(data1))  # 原始数据的索引
x_new = np.linspace(0, 1, len(data2))  # 新的数据索引

# 使用样条插值进行重采样
spline = interp1d(x_original, data1.values.flatten(), kind='cubic')  # 使用立方样条插值
data1_resampled = spline(x_new)

# 保存重采样后的数据
output_path = r'C:\Users\陈宇飞\Desktop\华为杯\重采样后的声音信号.txt'
pd.DataFrame(data1_resampled).to_csv(output_path, header=False, index=False)

# 绘制振幅-时间图
plt.figure(figsize=(10, 6))
plt.plot(range(len(data1_resampled)), data1_resampled, label="Resampled Signal", color="b")
plt.xlabel("Time (Index)")
plt.ylabel("Amplitude")
plt.title("Amplitude-Time Plot of Resampled Signal")
plt.legend()
plt.grid(True)
plt.savefig('resampled.png')
#plt.show()



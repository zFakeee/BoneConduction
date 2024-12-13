import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# 读取数据
amplitude = np.loadtxt(r'C:\Users\陈宇飞\Desktop\华为杯\串口数据.txt')  # 请将 'data.txt' 替换为你的文件名

# 生成时间数据
time = np.arange(len(amplitude))  # 使用数据的索引作为时间

# 低通滤波函数
def lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# 设置参数
cutoff_frequency = 0.5  # 低通截止频率
sampling_rate = 5.0  # 采样频率 (根据你的数据进行调整)

# 进行低通滤波
filtered_amplitude = lowpass_filter(amplitude, cutoff_frequency, sampling_rate)

# 将低通滤波后的数据保存到新文件
output_file_path = r'C:\Users\陈宇飞\Desktop\华为杯\低通滤波后的数据.txt'
np.savetxt(output_file_path, filtered_amplitude, fmt='%f')

# 输出保存文件的路径
print("低通滤波后的数据已保存到:", output_file_path)

# 绘制时间振幅图
plt.figure(figsize=(12, 6))
plt.plot(time, amplitude, label='Original amplitude', alpha=0.5)
plt.plot(time, filtered_amplitude, label='Amplitude after low-pass filtering', color='red')
plt.xlabel('Sample index')
plt.ylabel('Amplitude')
plt.title('Time amplitude plot')
plt.legend()
plt.grid()
plt.savefig('filter.png')
#plt.show()

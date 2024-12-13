import numpy as np
import matplotlib.pyplot as plt

def read_signal_from_file(file_path):
    with open(file_path, 'r') as file:
        signal = np.array([float(line.strip()) for line in file if line.strip()])
    return signal

def write_signal_to_file(signal, file_path):
    with open(file_path, 'w') as file:
        for value in signal:
            file.write(f"{value}\n")

def lms_filter(desired_signal, input_signal, filter_length, mu):
    n_samples = len(input_signal)
    weights = np.zeros(filter_length)
    output_signal = np.zeros(n_samples)
    error_signal = np.zeros(n_samples)

    for n in range(n_samples):
        x = np.zeros(filter_length)
        x[:min(filter_length, n+1)] = input_signal[max(0, n-filter_length+1):n+1][::-1]

        output_signal[n] = np.dot(weights, x)
        error_signal[n] = desired_signal[n] - output_signal[n]
        weights += 2 * mu * error_signal[n] * x

    return output_signal, error_signal, weights

def signal_recovery(estimated_signal, estimated_channel_weights):
    recovered_signal = np.convolve(estimated_signal, estimated_channel_weights, mode='same')
    return recovered_signal

# 主程序
initial_signal_file = r"C:\Users\陈宇飞\Desktop\华为杯\原始数据.txt"
final_signal_file = r"C:\Users\陈宇飞\Desktop\华为杯\原始振动数据.txt"
estimated_signal_file = r"C:\Users\陈宇飞\Desktop\华为杯\重采样后的声音信号.txt"
recovered_signal_file = r"C:\Users\陈宇飞\Desktop\华为杯\估计后信号.txt"

initial_signal = read_signal_from_file(initial_signal_file)
final_signal = read_signal_from_file(final_signal_file)

filter_length = 10  # 可以根据需要调整滤波器长度
mu = 0.01  # 学习率

_, _, estimated_channel = lms_filter(final_signal, initial_signal, filter_length, mu)

estimated_signal = read_signal_from_file(estimated_signal_file)
recovered_signal = signal_recovery(estimated_signal, estimated_channel)

write_signal_to_file(recovered_signal, recovered_signal_file)

print("复原的信号已保存到:", recovered_signal_file)

# 绘制振幅-时间图
plt.figure(figsize=(10, 6))

# 绘制重采样后的声音信号
plt.plot(range(len(estimated_signal)), estimated_signal, label="Resampled Sound Signal", color="b")

# 绘制估计后的信号
plt.plot(range(len(recovered_signal)), recovered_signal, label="Recovered Signal", color="r")

# 设置图形标签和标题
plt.xlabel("Time (Index)")
plt.ylabel("Amplitude")
plt.title("Amplitude-Time Plot of Resampled Sound Signal and Recovered Signal")
plt.legend()
plt.grid(True)
plt.savefig('estimated.png')
#plt.show()


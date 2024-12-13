import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Step 1: Load the WAV file
wav_file = r'C:\Users\陈宇飞\Desktop\华为杯\华为杯.wav'  # 替换为你的WAV文件路径
sample_rate, data = wavfile.read(wav_file)

# Step 2: Create time axis for plotting
time = np.linspace(0, len(data) / sample_rate, num=len(data))

# Step 3: Plot the waveform
plt.figure(figsize=(15, 5))
plt.plot(time, data)
plt.title('Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid()
plt.savefig('waveform.png')  # 保存波形图像
#plt.show()

# Step 4: Save data to TXT file
np.savetxt(r'C:\Users\陈宇飞\Desktop\华为杯\huawei.txt', data)


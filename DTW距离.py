from fastdtw import fastdtw
import numpy as np
import os

# 读取数据并归一化
def load_and_normalize_data(file_path):
    data = np.loadtxt(file_path)
    mean = np.mean(data)
    std_dev = np.std(data)
    normalized_data = (data - mean) / std_dev
    return normalized_data


# 计算DTW距离
def calculate_dtw(file1, file2):
    data1 = load_and_normalize_data(file1)
    data2 = load_and_normalize_data(file2)

    distance, path = fastdtw(data1, data2)
    print(distance)
    return distance, path


def check_pairing(file1, file2):
    # 检查是否是第一次运行
    marker_file = r'C:\Users\陈宇飞\Desktop\华为杯\first_run.txt'
    dtw_distance, dtw_path = calculate_dtw(file1, file2)
    result_message = f"一致性计算结果为: {dtw_distance}\n"
    if os.path.exists(marker_file):
        # 如果标记文件存在，表示已配对过，返回失败
        result_message += "配对失败"
        return result_message
    else:
        # 如果标记文件不存在，表示第一次运行，进行配对并创建标记文件
        with open(marker_file, 'w') as f:
                f.write("This is the first run, pairing successful.\n")
        if dtw_distance < 160:
                result_message += "配对成功"
        else:
                result_message += "配对失败"

        return result_message








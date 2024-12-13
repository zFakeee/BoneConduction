import torch
import numpy as np
import torch.nn.functional as F

from train import CNNModel  # 假设模型代码在model.py中


# 加载模型
def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 切换为评估模式
    print(f"Model loaded from {model_path}")
    return model


# 对输入数据进行预测
def predict(model, data):
    # 对数据进行必要的预处理，确保长度为9999
    if len(data) < 9999:
        data = np.pad(data, (0, 9999 - len(data)), 'constant')
    else:
        data = data[:9999]

    # 将数据转换为 Tensor，并确保形状为 [1, 1, 9999]（batch_size=1, channels=1, sequence_length=9999）
    data = torch.tensor(data, dtype=torch.float32)  # 转为Tensor
    data = data.unsqueeze(0)  # 添加批次维度和通道维度，形状变为 [1, 1, 9999]

    # 推断
    with torch.no_grad():
        output = model(data)
        # 使用softmax计算各类的概率
        probabilities = F.softmax(output, dim=1)
        predicted = torch.max(output.data, 1)

    predicted_label = predicted.indices.item()  # 获取预测标签

    # 获取配对成功和失败的概率
    pair_success_prob = probabilities[0][1].item()  # 假设配对成功对应的类标签是1
    pair_fail_prob = probabilities[0][0].item()  # 假设配对失败对应的类标签是0

    # 输出配对结果和相似度
    if predicted_label == 1:
        print(f"配对成功, 相似度: {pair_success_prob:.4f}")
    elif predicted_label == 0:
        print(f"配对失败, 相似度: {1-pair_fail_prob:.4f}")

    return predicted_label, pair_success_prob, pair_fail_prob


# 从txt文件读取数据并转换为npy格式
def load_data_from_txt(txt_file_path):
    # 读取txt文件中的数据，假设每行一个数值
    data = np.loadtxt(txt_file_path)
    return data


if __name__ == '__main__':
    # 加载模型
    model = CNNModel(num_classes=5)
    model = load_model(model, 'cnn_model.pth')  # 加载训练好的模型

    # 读取txt文件
    txt_file_path = r'C:\Users\陈宇飞\Desktop\DNN\4\filtered_3.txt'  # 替换为实际的txt文件路径
    data = load_data_from_txt(txt_file_path)

    # 预测
    predicted_label, pair_success_prob, pair_fail_prob = predict(model, data)
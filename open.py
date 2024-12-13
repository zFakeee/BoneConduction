from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5 import QtGui
import sys
import os
from PyQt5 import uic
from PyQt5.QtCore import Qt
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from main import Ui_MainWindow
import subprocess
from fastdtw import fastdtw
import serial
import threading
from DTW距离 import check_pairing
import torch
import torch.nn.functional as F
from train import CNNModel
# Global variables
is_collecting = False
data_buffer = []
serial_port = None

def collect_serial_data():
    global is_collecting, data_buffer, serial_port
    while is_collecting:
        try:
            if serial_port.in_waiting > 0:
                data = serial_port.readline().decode('utf-8').strip()
                data_buffer.append(data)
                ui.dataTextEdit.append(data)
        except Exception as e:
            print(f"Error reading from serial: {e}")

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    print(f"Model loaded from {model_path}")
    return model

# Function to predict with the model
def predict(model, data):
    if len(data) < 9999:
        data = np.pad(data, (0, 9999 - len(data)), 'constant')
    else:
        data = data[:9999]

    data = torch.tensor(data, dtype=torch.float32)
    data = data.unsqueeze(0)  # Add batch and channel dimensions

    with torch.no_grad():
        output = model(data)
        probabilities = F.softmax(output, dim=1)
        predicted = torch.max(output.data, 1)

    predicted_label = predicted.indices.item()
    pair_success_prob = probabilities[0][1].item()  # Success probability
    pair_fail_prob = probabilities[0][0].item()  # Failure probability

    result_message = ""
    if predicted_label == 1:
        result_message = f"配对成功, 相似度: {pair_success_prob:.4f}"
    else:
        result_message = f"配对失败, 相似度: {1 - pair_fail_prob:.4f}"

    return result_message


def start_collection():
    global is_collecting, data_buffer, serial_port
    if not is_collecting:
        with open(r'C:\Users\陈宇飞\Desktop\华为杯\串口数据.txt', 'w') as f:
            f.truncate(0)  # 清空文件内容
        # Open the serial port
        serial_port = serial.Serial('COM3', 115200, timeout=1)  # Adjust the port and baud rate as needed
        is_collecting = True
        data_buffer = []  # Clear previous data
        threading.Thread(target=collect_serial_data, daemon=True).start()
        ui.gatherpushButton.setText("停止采集")
    else:
        # Stop collecting data
        is_collecting = False
        serial_port.close()  # Close the serial port
        ui.gatherpushButton.setText("开始采集")

        # Save the collected data to a txt file
        with open(r'C:\Users\陈宇飞\Desktop\华为杯\串口数据.txt', 'w') as f:
            for line in data_buffer:
                f.write(f"{line}\n")
        print("Data saved to collected_data.txt")
def open_script1():
    subprocess.Popen(['python', '转wav格式并输出数据.py'])
    ui.label_3.setPixmap(QtGui.QPixmap('waveform.png'))  # 加载图像到 QLabel
    ui.label_3.setScaledContents(True)  # 允许缩放内容
def open_script2():
    ui.dataTextEdit.clear()
    ui.label_3.raise_()
    subprocess.Popen(['python', '低通滤波.py'])
    ui.label_3.setPixmap(QtGui.QPixmap('filter.png'))  # 加载图像到 QLabel
    ui.label_3.setScaledContents(True)
def open_script3():
    subprocess.Popen(['python', '重采样.py'])
    ui.label_3.setPixmap(QtGui.QPixmap('resampled.png'))  # 加载图像到 QLabel
    ui.label_3.setScaledContents(True)
def open_script4():
    subprocess.Popen(['python', '信道估计.py'])
    ui.label_3.setPixmap(QtGui.QPixmap('estimated.png'))  # 加载图像到 QLabel
    ui.label_3.setScaledContents(True)
def open_script5():
    ui.label_3.clear()  # 清空 label_3 的内容
    ui.pairtextEdit.raise_()
    model = CNNModel(num_classes=5)
    model = load_model(model, 'cnn_model.pth')  # Load the pre-trained model
    # Load data from txt file
    txt_file_path = r'C:\Users\陈宇飞\Desktop\华为杯\低通滤波后的数据.txt'
    data = np.loadtxt(txt_file_path)
    # Get prediction results
    result_message = predict(model, data)
    ui.pairtextEdit.setPlainText(result_message)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    ui.samplingButton.clicked.connect(open_script1)
    MainWindow.setWindowOpacity(1)
    MainWindow.setWindowFlags(Qt.FramelessWindowHint)
    MainWindow.setAttribute(Qt.WA_TranslucentBackground)
    # 其他按钮连接...
    ui.gatherpushButton.clicked.connect(start_collection)
    ui.filteringButton_2.clicked.connect(open_script2)
    ui.ResamplingButton_3.clicked.connect(open_script3)
    ui.estimationButton_5.clicked.connect(open_script4)
    ui.pairingButton_4.clicked.connect(open_script5)
    MainWindow.show()
    sys.exit(app.exec_())




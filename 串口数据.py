import serial
import time

# 设置串口参数
port = 'COM3'  # 替换为你的串口号
baudrate = 115200  # 根据你的设备设置波特率
timeout = 1  # 读取超时设置（秒）

# 创建串口对象
ser = serial.Serial(port, baudrate, timeout=timeout)

# 打开文件以保存数据
with open(r'C:\Users\陈宇飞\Desktop\华为杯\串口数据.txt', 'a') as f:
    print("开始读取数据...（按 Ctrl+C 停止）")
    try:
        while True:
            # 从串口读取数据
            line = ser.readline().decode('utf-8').rstrip()  # 读取一行数据并去掉换行符
            if line:
                print(line)  # 打印读取的数据
                f.write(line + '\n')  # 保存数据到文件
            time.sleep(0.1)  # 可调整读取间隔
    except KeyboardInterrupt:
        print("停止读取数据。")
    finally:
        ser.close()  # 关闭串口

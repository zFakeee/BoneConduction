import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []

        # 遍历所有子文件夹，将样本和对应的标签存入列表
        for label_name in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label_name)
            if os.path.isdir(label_path):
                for file_name in os.listdir(label_path):
                    file_path = os.path.join(label_path, file_name)
                    if file_name.endswith('.npy'):
                        self.data.append(file_path)
                        # 假设文件名格式为 '1_sample.npy'，提取数字并减去1
                        label = int(label_name)
                        self.labels.append(label)

        # 将类名转换为数值标签（从0开始）
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(set(self.labels)))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data[idx]
        label_name = self.labels[idx]
        label = self.label_to_idx[label_name]

        # 加载样本数据 (假设为.npy格式)
        sample = np.load(file_path)

        # 检查样本是否为空
        if sample.size == 0:
            print(f"Warning: {file_path} is empty. Skipping this sample.")
            return None  # 直接返回None以在collate_fn中过滤

        # 确保样本长度为9999
        if len(sample) < 9999:
            padded_sample = np.pad(sample, (0, 9999 - len(sample)), 'constant')
        else:
            padded_sample = sample[:9999]

        # 转换为Tensor
        sample = torch.tensor(padded_sample, dtype=torch.float32)

        return sample, label

def collate_fn(batch):
    """
    自定义 collate_fn，过滤掉空样本，并将样本堆叠到一起。
    """
    # 过滤掉None样本
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:  # 如果批次为空，则返回空张量和空标签
        return torch.tensor([]), torch.tensor([])

    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    data = torch.stack(data)
    labels = torch.tensor(labels)

    return data, labels


class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # 添加Dropout层
        self.dropout_conv = nn.Dropout(0.5)  # 在卷积层后随机丢弃50%的神经元
        self.dropout_fc = nn.Dropout(0.5)  # 在全连接层前随机丢弃50%的神经元

        self.fc_input_dim = 128 * (9999 // 2 // 2 // 2)  # 经过3次池化
        self.fc1 = nn.Linear(self.fc_input_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        #print(x.shape)
        x = x.unsqueeze(1)
        #x = x.transpose(1, 2)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout_conv(x)  # 添加Dropout层
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout_conv(x)  # 添加Dropout层
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout_conv(x)  # 添加Dropout层
        x = x.view(-1, self.fc_input_dim)
        x = self.dropout_fc(F.relu(self.fc1(x)))  # 在全连接层前添加Dropout
        x = self.fc2(x)
        return x


def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        total, correct = 0, 0  # 用于计算训练准确率

        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()  # 清空梯度

            # 添加通道维度
            outputs = model(batch_data)  # batch_data 已经有正确的形状
            loss = criterion(outputs, batch_labels)
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            running_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

        accuracy = correct / total  # 计算训练准确率
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Training Accuracy: {accuracy:.4f}")

def evaluate_model(model, test_loader):
    model.eval()  # 设置模型为评估模式
    total, correct = 0, 0
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            outputs = model(batch_data)  # 不需要unsqueeze
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')

def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == '__main__':
    train_dir = "train/"  # 替换为实际的训练数据目录
    test_dir = "test/"  # 替换为实际的测试数据目录

    # 创建数据集实例
    train_dataset = CustomDataset(train_dir)
    test_dataset = CustomDataset(test_dir)

    # 使用 DataLoader 加载数据
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    num_classes = 5
    model = CNNModel(num_classes)

    # 创建损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train_model(model, train_loader, criterion, optimizer, num_epochs=20)

    save_model(model, 'cnn_model.pth')  # 保存训练好的模型
    # 测试模型
    evaluate_model(model, test_loader)
import torch
from torchvision import datasets, transforms
from torch.utils import data
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt  # 用于可视化
import numpy as np

"""读取数据集"""
# 定义数据变换（标准化处理）
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为tensor，并将像素值从0-255归一化到0-1
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
])

# 加载本地数据集
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=False,
    transform=transform
)
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=False,
    transform=transform
)

# 创建数据加载器（按批次加载数据）
train_loader = data.DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True
)
test_loader = data.DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False
)

"""定义LeNet模型"""


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.flatten = nn.Flatten()
        self._initialize_weights()  # 权重初始化

    def _initialize_weights(self):
        for m in self.modules():  # 遍历模型所有层
            if isinstance(m, (nn.Linear, nn.Conv2d)):  # 对线性层和卷积层初始化
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = F.avg_pool2d(F.relu(self.conv1(x)), 2)  # 输出(batch, 6, 14, 14)
        x = F.avg_pool2d(F.relu(self.conv2(x)), 2)  # 输出（batch,16,5,5)
        x = self.flatten(x)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x


# 检测GPU设备，优先使用GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

"""初始化模型，损失函数，优化器"""
net = LeNet()
net = net.to(device)
criterion = nn.CrossEntropyLoss()  # 损失函数
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

"""训练模型（带损失记录）"""
# 新增：用于记录损失的列表
train_losses = []  # 记录每轮训练的平均损失
test_losses = []  # 记录每轮测试的平均损失
epochs = 20

for epoch in range(epochs):
    net.train()  # 训练模式
    running_loss = 0.0

    # 训练过程
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()  # 清空梯度
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        running_loss += loss.item()

    # 计算本轮训练的平均损失并记录
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f'Epoch [{epoch + 1}/{epochs}], 训练平均损失: {avg_train_loss:.4f}')

    # 测试过程（计算测试集损失）
    net.eval()  # 评估模式
    test_running_loss = 0.0
    with torch.no_grad():  # 关闭梯度计算
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            test_running_loss += loss.item()

    # 计算本轮测试的平均损失并记录
    avg_test_loss = test_running_loss / len(test_loader)
    test_losses.append(avg_test_loss)
    print(f'Epoch [{epoch + 1}/{epochs}], 测试平均损失: {avg_test_loss:.4f}\n')

"""模型评估（准确率）"""
net.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Test Accuracy: {100 * correct / total:.2f}%')

"""可视化训练/测试损失曲线"""
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # Windows默认中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常
plt.figure(figsize=(10, 6))  # 设置图像大小
plt.plot(range(1, epochs + 1), train_losses, label='训练损失', marker='o', color='blue')
plt.plot(range(1, epochs + 1), test_losses, label='测试损失', marker='s', color='red')

# 添加标题和标签
plt.title('LeNet训练与测试损失曲线', fontsize=14)
plt.xlabel('Epoch（轮次）', fontsize=12)
plt.ylabel('损失值', fontsize=12)
plt.legend()  # 显示图例
plt.grid(linestyle='--', alpha=0.7)  # 添加网格线
plt.xticks(np.arange(1, epochs + 1, step=1))  # 设置x轴刻度
plt.show()  # 显示图像




import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

"""1. 数据加载模块"""


def load_data(batch_size, resize=224):
    """
    加载FashionMNIST数据集
    batch_size: 批次大小
    resize: 图像缩放尺寸，默认224（适配AlexNet）
    """
    # 数据变换：缩放→转张量→标准化
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化到[-1, 1]范围
    ])

    # 加载数据集
    train_dataset = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=False,  # 已下载设为False
        transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=False,
        transform=transform
    )

    # 创建数据加载器
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # 避免Windows多线程问题
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, test_loader


"""2. AlexNet模型定义"""


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # 卷积层部分
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),  # 输入1通道（灰度图）
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # 全连接层部分
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6400, 4096),  # 6400 = 256*5*5（224输入经卷积后的特征图尺寸）
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 10)
        )

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier初始化卷积层和全连接层权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


"""3. 模型训练与评估"""

def train_model(net, train_loader, test_loader, epochs=20, lr=0.01, device="cuda"):
    """
    训练模型并可视化损失曲线
    net: 网络模型
    train_loader/test_loader: 数据加载器
    epochs: 训练轮次
    lr: 学习率
    device: 训练设备
    """
    # 设备配置
    net = net.to(device)
    print(f"使用设备: {device}")

    # 损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)  # 添加动量加速收敛

    # 记录损失
    train_losses = []
    test_losses = []

    # 训练循环
    for epoch in range(epochs):
        # 训练阶段
        net.train()
        train_running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = net(images)
            loss = criterion(outputs, labels)

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()


        # 计算本轮训练平均损失
        avg_train_loss = train_running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f'Epoch [{epoch + 1}/{epochs}], 训练平均损失: {avg_train_loss:.4f}')

        # 测试阶段（计算测试损失）
        net.eval()
        test_running_loss = 0.0

        with torch.no_grad():  # 关闭梯度计算
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                test_loss = criterion(outputs, labels)
                test_running_loss += test_loss.item()

        avg_test_loss = test_running_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        print(f"Epoch [{epoch + 1}/{epochs}], 测试平均损失: {avg_test_loss:.4f}\n")

    # 计算测试集准确率
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

    accuracy = 100 * correct / total
    print(f"测试集准确率: {accuracy:.2f}%")

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label="训练损失", marker="o", color="blue")
    plt.plot(range(1, epochs + 1), test_losses, label="测试损失", marker="s", color="red")

    plt.title("AlexNet训练与测试损失曲线", fontsize=14)
    plt.xlabel("训练轮次 (Epoch)", fontsize=12)
    plt.ylabel("交叉熵损失", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(linestyle="--", alpha=0.7)
    plt.xticks(np.arange(1, epochs + 1, step=1))
    plt.tight_layout()
    plt.show()

    return net


"""4. 主函数执行"""
if __name__ == "__main__":
    # 超参数设置
    BATCH_SIZE = 128
    EPOCHS = 20
    LEARNING_RATE = 0.01

    # 加载数据
    train_loader, test_loader = load_data(batch_size=BATCH_SIZE)

    # 设备选择
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 实例化模型并训练
    model = AlexNet()
    trained_model = train_model(
        model,
        train_loader,
        test_loader,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        device=device
    )

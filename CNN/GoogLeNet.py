import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils import data

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

"""1.读取数据集"""
def read_data(batchsize,resize=None):
    """
        加载FashionMNIST数据集
        batch_size: 批次大小
        resize: 图像缩放尺寸，默认224（适配AlexNet）
        """
    #数据变换：缩放→转张量→标准化
    trans=[]
    if resize:
        trans.append(transforms.Resize(resize))
    trans.append(transforms.ToTensor())
    trans.append(transforms.Normalize([0.5],[0.5]))
    transform=transforms.Compose(trans)
    #读取数据集本地
    train_data=datasets.FashionMNIST(root='./data',train=True,transform=transform,download=False)
    test_data=datasets.FashionMNIST(root='./data',train=False,transform=transform,download=False)
    #dataloader
    return(data.DataLoader(dataset=train_data,batch_size=batchsize,shuffle=True,num_workers=0),
           data.DataLoader(dataset=test_data,batch_size=batchsize,shuffle=False,num_workers=0))

"""2.定义VGG模型"""
class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        #定义5个vgg块
        self.vgg1=self.vgg(1,1,8)
        self.vgg2=self.vgg(1,8,16)
        self.vgg3=self.vgg(2,16,32)
        self.vgg4=self.vgg(2,32,64)
        self.vgg5=self.vgg(2,64,64)
        #全连接层
        self.lin1=nn.Linear(64*7*7,4096)
        self.lin2=nn.Linear(4096,4096)
        self.lin3=nn.Linear(4096,10)
        self._initialize_weights()  #展平层
        self.flatten=nn.Flatten()    #权重初始化

    #定义一个VGG块
    def vgg(self,number_conv,in_channels,out_channels):
        layers= []
        for i in range(number_conv):
            layers.append(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1))
            layers.append(nn.ReLU())
            in_channels=out_channels
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():  # 遍历模型所有层
            if isinstance(m, (nn.Linear, nn.Conv2d)):  # 对线性层和卷积层初始化
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:  # 偏置初始化为0
                    nn.init.constant_(m.bias, 0)

    def forward(self,x):
        #依次通过5个VGG块
        x=self.vgg1(x)
        x=self.vgg2(x)
        x=self.vgg3(x)
        x=self.vgg4(x)
        x=self.vgg5(x)
        x=self.flatten(x)
        x=F.dropout(F.relu(self.lin1(x)),p=0.5)
        x=F.dropout(F.relu(self.lin2(x)),p=0.5)
        x=self.lin3(x)
        return x


"""3. 模型训练与评"""
def training(net,train_loader,test_loader,epochs,lr,device):
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
    # 损失函数与优化器
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr,momentum=0.9)
    # 记录损失
    train_losses = []
    test_losses = []
    #训练循环
    for epoch in range(epochs):
        net.train()
        running_loss=0.0
        for i,(x,y) in enumerate(train_loader):
            x,y=x.to(device),y.to(device)
            #前向传播
            y1=net(x)
            l=loss(y1,y)
            #反向传播与优化
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            running_loss += l.item()
        # 计算本轮训练平均损失
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f'Epoch [{epoch + 1}/{epochs}], 训练平均损失: {avg_train_loss:.4f}')

        # 测试阶段（计算测试损失）
        net.eval()
        test_running_loss = 0.0

        with torch.no_grad():  # 关闭梯度计算
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                test_loss = loss(outputs, labels)
                test_running_loss += test_loss.item()

        avg_test_loss = test_running_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        print(f"Epoch [{epoch + 1}/{epochs}], 测试平均损失: {avg_test_loss:.4f}\n")
    #评估模型
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # 关闭梯度计算，节省内存
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)  # 取概率最大的类别
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"测试集准确率: {100 * correct / total:.2f}%")

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


if __name__ == "__main__":
    # 超参数设置
    batchsize = 128
    epochs = 20
    lr = 0.01
    # 读取数据集
    train_loader, test_loader = read_data(128, 224)
    #设备选择
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #模型实例化并训练
    net = VGG()
    trained_model=training(net,
                           train_loader,
                           test_loader,
                           epochs=epochs,
                           lr=lr,
                           device=device)





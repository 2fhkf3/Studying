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

"""读取数据集"""
def read_data(batchsize,resize=None):
    """
           加载FashionMNIST数据集
           batch_size: 批次大小
           resize: 图像缩放尺寸，默认224（适配AlexNet）
           """
    # 数据变换：缩放→转张量→标准化
    trans= []
    if resize:
        trans.append(transforms.Resize(resize))
    trans.append(transforms.ToTensor())
    trans.append(transforms.Normalize([0.5],[0.5]))
    transform=transforms.Compose(trans)

    #读取数据集
    train_data=datasets.FashionMNIST(root='./data',train=True,transform=transform,download=False)
    test_data=datasets.FashionMNIST(root='./data',train=False,transform=transform,download=False)

    #返回迭代器
    x=data.DataLoader(dataset=train_data,batch_size=batchsize,shuffle=True,num_workers=0)
    y=data.DataLoader(dataset=test_data,batch_size=batchsize,shuffle=False,num_workers=0)

    return x,y

"""定义NiN模型"""
class NiN(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature=nn.Sequential(
            self.nin(1,96,11,4,0),
            nn.MaxPool2d(3,2),
            self.nin(96,256,5,1,2),
            nn.MaxPool2d(3,2),
            self.nin(256,384,3,1,1),
            nn.MaxPool2d(3,2),
            nn.Dropout(0.5),
            self.nin(384,10,3,1,1),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )
        self._initialize_weights()

    #定义一个NiN块
    def nin(self,in_channels,out_channels,kernel_size,strides,padding):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=strides,padding=padding),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size=1),
            nn.ReLU()
        )

    def _initialize_weights(self):
        for m in self.modules():  # 遍历模型所有层
              if isinstance(m, (nn.Linear, nn.Conv2d)):  # 对线性层和卷积层初始化
                  nn.init.xavier_uniform_(m.weight)
                  if m.bias is not None:  # 偏置初始化为0
                       nn.init.constant_(m.bias, 0)


    def forward(self,x, print_layer_shape=False):
        if print_layer_shape:
            print(f"初始输入形状: {x.shape}")  # 打印输入尺寸
            for idx, layer in enumerate(self.feature, 1):  # 遍历每层，带序号
                x = layer(x)  # 层计算
                # 打印：层序号、层名称、当前输出形状
                print(f"第{idx:2d}层 ({layer.__class__.__name__:12s}) → 输出形状: {x.shape}")
        else:
            x = self.feature(x)  # 正常训练时不打印
        return x

"""3. 模型训练与评估"""
def data_training(net,train_loader,test_loader,epochs,lr,device):
    """
            训练模型并可视化损失曲线
            net: 网络模型
            train_loader/test_loader: 数据加载器
            epochs: 训练轮次
            lr: 学习率
            device: 训练设备
            """
    #配置设备
    net=net.to(device)
    #损失函数，优化器
    loss=nn.CrossEntropyLoss()  #交叉熵损失函数
    optimizer=torch.optim.SGD(net.parameters(),lr=lr)  #SGD优化器
    # 记录损失
    train_losses = []
    test_losses = []
    # 训练循环
    for epoch in range(epochs):
        net.train()
        running_loss=0.0
        for i,(image,label) in enumerate(train_loader):
            image,label=image.to(device),label.to(device)
            #前向传播
            label_y=net(image)
            l=loss(label_y,label)
            #反向传播优化参数
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            #计算损失函数值
            running_loss+=l.item()
            # 计算本轮训练平均损失
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f'Epoch [{epoch + 1}/{epochs}], 训练平均损失: {avg_train_loss:.4f}')


        #测试阶段（计算测试损失）
        net.eval()
        test_running_loss = 0.0
        with torch.no_grad():
            for images,labels in test_loader:
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
            images,labels= images.to(device),labels.to(device)
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
    lr = 0.1
    # 读取数据集
    train_loader, test_loader = read_data(128, 224)
    #设备选择
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #模型实例化并训练
    net = NiN()
    # 测试并打印每层输出大小（修正后）
    print("\n===== 每层输出大小 =====")
    test_input = torch.randn(1, 1, 224, 224)
    output = net(test_input, print_layer_shape=True)
    print(f"最终输出类型: {type(output)}")  # 应打印 <class 'torch.Tensor'>
    print("=======================\n")
    trained_model=data_training(net,
                           train_loader,
                           test_loader,
                           epochs=epochs,
                           lr=lr,
                           device=device)
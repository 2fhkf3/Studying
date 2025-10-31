import collections
import math
import re
from torch.nn import functional as F
import torch
from d2l import torch as d2l
import random
from torch import nn



# 定义批量大小和时间步数
batch_size, num_steps = 32, 35

"""读取数据"""
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                               '090b5e7e70c295757f55df93cb0a180b9691891a')

#将文本读入，按行存入lines中，同时只保留字母（大写改为小写），空格
def read_time_machine(line=None):
    with open(d2l.download('time_machine'),'r') as f:
        lines=f.readlines()
    return [re.sub('[^A-Za-z]+',' ',line).strip().lower() for line in lines]


#根据token指令分割类型
def tokenize(lines,token='word'):
    if token=='word':
        return [line.split() for line in lines]
    if token=='char':
        return [list(line) for line in lines]
    else:
        print('错位：未知令牌类型：'+token)


#词表类 用于构建词汇表，实现令牌（如单词、字符）与整数索引的双向映射，方便将文本转换为模型可处理的数值形式。
class Vocab:
    def __init__(self,tokens=None,min_freq=0,reserved_tokens=None):
       """文本词表"""
       if tokens is None:
           tokens=[]
       if reserved_tokens is None:
            reserved_tokens=[]
       counter=count_corpus(tokens)
       self.token_freqs=sorted(counter.items(),key=lambda x:int(x[1]),reverse=True)
       self.unk,uniq_tokens=0,['<unk>']+reserved_tokens
       uniq_tokens+=[token for token,freq in self.token_freqs if freq>=min_freq and token not in uniq_tokens]
       self.idx_to_token,self.token_to_idx=[],dict()
       for token in uniq_tokens:
           self.idx_to_token.append(token)
           self.token_to_idx[token]=len(self.idx_to_token)-1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self,tokens):
        if not isinstance(tokens,(list,tuple)):
            return self.token_to_idx.get(tokens,self.unk)
        return [self.__getitem__(token) for token in tokens]

#辅助函数：统计输入令牌tokens出现的频率
def count_corpus(tokens):
        """统计标记的频率"""
        if len(tokens)==0 or isinstance(tokens[0],list):
            tokens=[token for line in tokens for token in line]
        return collections.Counter(tokens)

#调用上面的函数整合数据 从 “时光机器” 数据集中提取文本，将其转换为字符级别的令牌（token），再映射为整数索引，同时构建对应的词汇表
def load_corpus_time_machine(max_token=-1):
    """返回时光机器数据据的标记索引列表和词汇表"""
    lines=read_time_machine()
    tokens=tokenize(lines,'char')
    vocab=Vocab(tokens)
    corpus=[vocab[token] for line in tokens for token in line]
    if max_token>0:
        corpus=corpus[:max_token]
    return corpus,vocab

#获取打乱的小批量子序列
def seq_data_iter_random(corpus,batch_size,num_steps):
    """使用随机抽样生成一个小批量子序列"""
    corpus=corpus[random.randint(0,num_steps-1):]
    #计算能够生成的子序列数量
    num_subseqs=(len(corpus)-1)//num_steps
    #创建初试索引列表
    initial_indices=list(range(0,num_subseqs*num_steps,num_steps))
    #进行随机打乱
    random.shuffle(initial_indices)
    #返回从指定位置开始的长度为num_steps的子序列
    def data(pos):
        return corpus[pos:pos+num_steps]
    #计算批次的数量
    num_batches=num_subseqs//batch_size
    #对每个批次进行迭代
    for i in range(0,batch_size*num_batches,batch_size):
        #获取当前批次的初始索引列表
        initial_indices_per_batch=initial_indices[i:i+batch_size]
        #根据初始索引列表获取对应的特征序列x
        x=[data(j) for j in initial_indices_per_batch]
        #根据初始索引列表获取对应的标签序列Y
        y=[data(j+1) for j in initial_indices_per_batch]
        yield torch.tensor(x),torch.tensor(y)


#获得顺序的小批量子序列
def seq_data_iter_sequential(corpus,batch_size,num_steps):
    """使用顺序分区生成一个小批量子序列"""
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset:offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1:offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    # 对每个批次进行迭代
    for i in range(0, num_steps * num_batches, num_steps):
        # 获取当前批次的特征序列X
        X = Xs[:, i:i + num_steps]
        # 获取当前批次的标签序列Y
        Y = Ys[:, i:i + num_steps]
        # 使用yield语句返回X和Y作为生成器的输出
        yield X, Y

#构建 “时光机器”（The Time Machine）文本数据集的迭代器
class SeDataLoader:
    def __init__(self,batch_size,num_steps,use_random_iter,max_tokens):
        if use_random_iter:
            #随机迭代
            self.data_iter_fn=seq_data_iter_random
        else:
            #顺序迭代
            self.data_iter_fn=seq_data_iter_sequential
        #corpus存“时光机器"的字符串的索引序号，语料库字符索引序列  vocab中存字典
        self.corpus,self.vocab=load_corpus_time_machine(max_tokens)
        self.batch_size,self.num_steps=batch_size,num_steps
    #使类的实例可迭代
    def __iter__(self):
        return self.data_iter_fn(self.corpus,self.batch_size,self.num_steps)
def load_time_machine(batch_size,num_steps,use_random_iter=False,max_tokens=10000):
    data_iter=SeDataLoader(batch_size,num_steps,use_random_iter,max_tokens)
    return data_iter,data_iter.vocab


"""初始化模型"""
# 初始化循环神经网络模型的模型参数
def get_params(vocab_size,num_hiddens,device):
    #设置输入和输出的维度为词汇表大小
    num_inputs=num_outputs=vocab_size
    #定义normal函数用于生成付出正态分布的随机张量，并乘以0.01进行缩放
    def normal(shape):
        return torch.randn(size=shape,device=device)*0.01
    #初始化核心参数
    #输入到隐藏层的权重矩阵，形状为（词汇表大小，隐藏单元个数）
    W_xh=normal((num_inputs,num_hiddens))
    #隐藏层到隐藏层的权重矩阵，形状为（隐藏单元个数，隐藏单元个数）
    W_hh=normal((num_hiddens,num_hiddens))
    #隐藏层的偏执向量，形状为（隐藏单元个数）
    b_h=torch.zeros(num_hiddens,device=device)
    #隐藏层到输出层的权重矩阵，形状为（隐藏单元个数，词汇表大小）
    W_hq=normal((num_hiddens,num_outputs))
    #输出层的偏置向量，形状为（词汇表大小）
    b_q=torch.zeros(num_outputs,device=device)
    #将所有参数放入列表中
    params=[W_xh,W_hh,b_h,W_hq,b_q]
    #遍历所有参数
    for param in params:
        #设置参数的requires_grad为True
        param.requires_grad_(True)
    return params

#在初始化时返回隐藏状态
def init_rnn_state(batch_size,num_hiddens,device):
    return (torch.zeros((batch_size,num_hiddens),device=device),)

#定义如何在一个时间步计算隐藏状态和输出  相当于forward
def rnn(inputs,state,params):
    W_xh,W_hh,b_h,W_hq,b_q=params
    H,=state
    outputs=[]
    for X in inputs:
        H=torch.tanh(torch.mm(X,W_xh)+torch.mm(H,W_hh)+b_h)
        Y=torch.mm(H,W_hq)+b_q
        outputs.append(Y)
    return torch.cat(outputs,dim=0),(H,)

#创建一个类来包装这些函数
class RNNModelScratch:
    #初始化模型参数
    def __init__(self,vocab_size,num_hiddens,device,get_params,init_state,forward_fn):
        #保存词汇表大小和隐藏单元个数作为类的属性
        self.vocab_szie,self.num_hiddens=vocab_size,num_hiddens
        #调用get_params函数初始化模型的参数，并保存为类的属性
        self.params=get_params(vocab_size,num_hiddens,device)
        #初始化隐藏状态的函数和前向传播函数
        self.init_state,self.forward_fn=init_state,forward_fn

    def __call__(self, X,state):
        X=F.one_hot(X.T,self.vocab_szie).type(torch.float32)
        return self.forward_fn(X,state,self.params)
    def begin_state(self,batch_size,device):
        #返回初始化的隐藏状态，用于模型的初始时间步
        return self.init_state(batch_size,self.num_hiddens,device)

# 检查输出是否具有正确的形状
# 设置隐藏单元个数为 512
train_iter, vocab = load_time_machine(batch_size, num_steps)
X = torch.arange(10).reshape((2,5))
num_hiddens = 512
# 创建一个 RNNModelScratch 的实例 net，指定词汇表大小、隐藏单元个数、设备、获取参数函数、初始化隐藏状态函数和前向传播函数
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                     init_rnn_state, rnn)
# 获取模型的初始隐藏状态，输入的批量大小为 X 的行数，设备使用与 X 相同的设备
state = net.begin_state(X.shape[0], d2l.try_gpu())
# 使用输入 X 和初始隐藏状态进行前向传播计算，得到输出张量 Y 和更新后的隐藏状态 new_state
# 将输入和状态都移动到与 X 相同的设备上进行计算
Y, new_state = net(X.to(d2l.try_gpu()),state)
# 输出 Y 的形状，new_state 的长度（即元素个数）和 new_state 中第一个元素的形状
print(Y.shape, len(new_state), new_state[0].shape)



#定义预测函数来生成用户提供的prefix之后的新字符
def predict_ch8(prefix,num_preds,net,vocab,device):
    """在’prefix‘后面生成新字符"""
    state=net.begin_state(batch_size=1,device=device)
    outputs=[vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape(1, 1)

    for y in prefix[1:]:
        _,state=net(get_input(),state)
        outputs.append(vocab[y])

    #生成指定数量的新字符
    for _ in range(num_preds):
        y,state=net(get_input(),state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
print(predict_ch8('time traveller ', 10, net, vocab, d2l.try_gpu()))

#梯度裁剪
def grad_clipping(net,theta):
    #如果net是nn.Module的实例
    if isinstance(net,nn.Module):
        params=[p for p in net.parameters() if p.requires_grad]
    else:
        params=net.params
    #计算参数梯度的范数，即所有参数梯度平方和的平方根
    norm=torch.sqrt(sum(torch.sum((p.grad**2))for p in params))
    #如果梯度范数超过指定阈值theta
    if norm > theta:
        for param in params:
            #将参数的梯度值裁剪至指定范围内，保持梯度范数不超过theta
            param.grad[:]*=theta/norm

#定义一个函数来训练只有一个迭代周期的模型
def train_epoch_ch8(net,train_iter,loss,updater,device,use_random_iter):
    state,timer=None,d2l.Timer()
    metric=d2l.Accumulator(2)
    for x,y in train_iter:
        if state is None or use_random_iter:
            #初始化隐藏状态，批量大小为x的行数，设备为指定的设备
            state=net.begin_state(batch_size=x.shape[0],device=device)
        else:
            if isinstance(net,nn.Module) and not isinstance(state,tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_()
        y=y.T.reshape(-1)
        x,y=x.to(device),y.to(device)
        y_hat,state=net(x,state)
        l=loss(y_hat,y.long()).mean()
        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net,1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)
        #累加损失和样本数量
        metric.add(l*y.numel(),y.numel())
    return math.exp(metric[0]/metric[1]),metric[1]/timer.stop()

# 训练函数支持从零开始或使用高级API实现的循环神经网络模型
def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    """训练模型"""
    #定义损失函数为加插上损失
    loss=nn.CrossEntropyLoss()
    # 创建动画对象，用于可视化训练过程的损失变化
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity', legend=['train'], xlim=[10, num_epochs])
    if isinstance(net, nn.Module):
        # 使用 PyTorch 的优化器 SGD 进行参数更新
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        # # 否则，使用自定义的梯度下降函数进行参数更新
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)

    predict=lambda prefix:predict_ch8(prefix, 50, net, vocab, device)
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter)
        # 每隔 10 个迭代周期生成
        if (epoch + 1) % 10 == 0:
            # 打印以 'time traveller' 为前缀的新字符序列
            print(predict('time traveller'))
            # 将当前迭代周期的困惑度添加到动画中进行可视化
            animator.add(epoch + 1, [ppl])
        # 打印最终的困惑度和每秒样本处理速度
        print(f'困惑度 {ppl:.1f}, {speed:.1f} 标记/秒 {str(device)}')
        # 生成并打印以 'time traveller' 为前缀的新字符序列
        print(predict('time traveller'))
        # 生成并打印以 'traveller' 为前缀的新字符序列
        print(predict('traveller'))
# 现在我们可以训练循环神经网络模型
# 设置迭代周期数和学习率
num_epochs, lr = 500, 1
# 调用训练函数进行模型训练，使用训练数据迭代器、词汇表、学习率、迭代周期数和设备信息作为输入
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())
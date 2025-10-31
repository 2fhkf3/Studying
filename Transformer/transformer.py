import os
from d2l import torch as d2l
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np
import copy
from torch.autograd import Variable

"""输入部分"""
class Embeddings(nn.Module):
    def __init__(self,d_model,vocab):
        super(Embeddings,self).__init__()
        self.lut=nn.Embedding(vocab,d_model)
        self.d_model=d_model
    def forward(self,x):
        return self.lut(x)*math.sqrt(self.d_model)
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout,max_len=5000):
        super(PositionalEncoding,self).__init__()
        self.dropout=nn.Dropout(p=dropout)
        pe=torch.zeros(max_len,d_model)
        position=torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,d_model,2)*-(math.log(1000.0)/d_model))
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        pe=pe.unsqueeze(0)
        self.register_buffer('pe',pe)
    def forward(self,x):
        x=x+self.pe[:, :x.size(1)]
        return self.dropout(x)


"""编码器"""
def subsequent_mask(size):
    attn_shape=(1,size,size)
    subsequent_mask=np.triu(np.ones(attn_shape),k=1).astype('uint8')
    return torch.from_numpy(1-subsequent_mask)
def attention(query,key,value,mask=None,dropout=None):
    d_k=query.size(-1)
    scores=torch.matmul(query,key.transpose(-2,-1)/math.sqrt(d_k))
    if mask is not None:
        scores=scores.masked_fill(mask==0,-1e9)
    p_attn=F.softmax(scores,dim=-1)
    if dropout is not None:
        p_attn=dropout(p_attn)
    return torch.matmul(p_attn,value),p_attn
def clones(module,N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N) ])
class MutiHeadAttention(nn.Module):
    def __init__(self,head,embedding_dim,dropout=0.1):
        super(MutiHeadAttention,self).__init__()
        assert embedding_dim%head==0
        self.d_k=embedding_dim//head
        self.head=head
        self.embedding_dim=embedding_dim

        self.W_q=nn.Linear(embedding_dim,embedding_dim)
        self.W_k = nn.Linear(embedding_dim, embedding_dim)
        self.W_v = nn.Linear(embedding_dim, embedding_dim)

        self.attn=None
        self.linear=nn.Linear(embedding_dim, embedding_dim)
        self.dropout=nn.Dropout(p=dropout)
        self.layernorn=nn.LayerNorm(embedding_dim)
    def forward(self,query,key,value,mask=None):
        residual=query
        batch_size=query.size(0)
        query=self.W_q(query).view(batch_size,-1,self.head,self.d_k).transpose(1,2)
        key = self.W_k(key).view(batch_size, -1, self.head, self.d_k).transpose(1,2)
        value=self.W_v(value).view(batch_size,-1,self.head,self.d_k).transpose(1,2)
        if mask is not None:
            mask=mask.unsqueeze(1).repeat(1,self.head,1,1)
        x,self.attn=attention(query,key,value,mask=mask,dropout=self.dropout)
        x=x.transpose(1,2).contiguous().view(batch_size,-1,self.head*self.d_k)
        output=self.linear(x)
        return   self.layernorn(residual+output)

class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super(PositionwiseFeedForward,self).__init__()
        self.w1=nn.Linear(d_model,d_ff)
        self.w2=nn.Linear(d_ff,d_model)
        self.dropout=nn.Dropout(p=dropout)
        self.layernorm=nn.LayerNorm(d_model)
    def forward(self,x):
        residual=x
        x=self.w2(self.dropout(F.relu(self.w1(x))))
        return self.layernorm(residual+x)

class EncoderLayer(nn.Module):
    def __init__(self,size,self_attn,feed_forward,dropout):
        super(EncoderLayer,self).__init__()
        self.self_attn=self_attn
        self.feed_forward=feed_forward
        self.size=size
        self.dropout=dropout
    def forward(self,x,mask):
        x=self.self_attn(x,x,x,mask)
        return self.feed_forward(x)

class Encoder(nn.Module):
    def __init__(self,layer,N):
        super(Encoder,self).__init__()
        self.layer=nn.ModuleList([layer for _ in range(N)])
        self.norm = nn.LayerNorm(layer.size)
    def forward(self,x,mask):
        for layer in self.layer:
            x=layer(x,mask)
        return self.norm(x)

"""解码器"""
class DecoderLayer(nn.Module):
    def __init__(self,size,self_attn,src_attn,feed_forward,dropout):
        super(DecoderLayer,self).__init__()
        self.size=size
        self.self_attn=self_attn
        self.src_attn=src_attn
        self.feed_forward=feed_forward
        self.dropout=dropout
    def forward(self,x,memory,source_mask,target_mask):
        m=memory
        x=self.self_attn(x,x,x,target_mask)
        x=self.src_attn(x,m,m,source_mask)
        return self.feed_forward(x)

class Decoder(nn.Module):
    def __init__(self,layer,N):
        super(Decoder,self).__init__()
        self.layer=nn.ModuleList([layer for _ in range(N)])
        self.norm = nn.LayerNorm(layer.size)
    def forward(self,x,memory,source_mask,target_mask):
        for layer in self.layer:
            x=layer(x,memory,source_mask,target_mask)
        return self.norm(x)

"""输出"""
class Generator(nn.Module):
    def __init__(self,d_model,vocab_size):
        super(Generator,self).__init__()
        self.project=nn.Linear(d_model,vocab_size)

    def forward(self,x):
        return F.log_softmax(self.project(x),dim=-1)

"""编码器解码器"""
class EncoderDecoder(nn.Module):
    def __init__(self,encoder,decoder,source_embed,target_embed,generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = source_embed
        self.tgt_embed = target_embed
        self.generator = generator
    def forward(self,source,target,source_mask,target_mask):
        memory=self.encoder(self.src_embed(source),source_mask)
        output=self.decoder(self.tgt_embed(target),memory,source_mask,target_mask)
        return self.generator(output)


def make_model(source_vocab,target_vocab,N=6,d_model=512,d_ff=2048,head=8,dropout=0.1):
    c=copy.deepcopy
    position= PositionalEncoding(d_model, dropout)  # 初始化位置编码输入
    attn=MutiHeadAttention(head,d_model,dropout)#初始化注意力层
    ffd=PositionwiseFeedForward(d_model,d_ff,dropout)#初始化前馈层
    model=EncoderDecoder(
        Encoder(EncoderLayer(d_model,c(attn),c(ffd),dropout),N),
        Decoder(DecoderLayer(d_model,c(attn),c(attn),c(ffd),dropout),N),
        nn.Sequential(Embeddings(d_model, source_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, target_vocab), c(position)),
        Generator(d_model,target_vocab)
    )
    for p in model.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)
    return model


"""读取数据"""
d2l.DATA_HUB['fra-eng']=(d2l.DATA_URL+'fra-eng.zip','94646ad1522d915e7b0f9296181140edcf86a4f5')

def read_data_nmt():
    """载入“英语-法语数据集"""
    #下载并解压数据集
    data_dir=d2l.download_extract('fra-eng')
    #读取数据并返回
    with open(os.path.join(data_dir,'fra.txt'),'r',encoding='utf-8')as f:
        return f.read()

#几个预处理步骤
def preprocess_nmt(text):
    """预处理“英语-法语”数据集"""
    #判断字符是否是特定标点符号并且前一个字符不是空格
    def no_space(char,prev_char):
        return char in set(',.!?')and prev_char !=' '
    #替换特殊字符为空格，转换为小写
    text=text.replace('\u202f',' ').replace('\xa0',' ').lower()
    out=[
        ' '+char if i>0 and no_space(char,text[i-1]) else char
        for i,char in enumerate(text)]
    return ''.join(out)

#词元化
def tokenize_nmt(text,num_examples=None):
    """词元化数据集"""
    #存储英语和法语的词元序列
    source,target=[],[]
    #遍历文本中的每一行
    for i ,line in enumerate(text.split('\n')):
        #如果指定了num_examples且超过了指定数量，则结束循环
        if num_examples and i>num_examples:
            break
        #按制表符分割行
        parts=line.split('\t')
        #如果行中包含了两个部分
        if len(parts)==2:
            # 将英语部分按空格分割为词元，并添加到source列表
            source.append(parts[0].split(' '))  # 英语
            # 将法语部分按空格分割为词元，并添加到target列表
            target.append(parts[1].split(' '))  # 法语
    return source,target
#调用函数词元化文本

#打印source和target的前6个词元序列



# 序列样本都有一个固定长度截断或填充文本序列
def truncate_pad(line,num_steps,padding_token):
    """阶段或填充文本序列"""
    #如果文本序列长度超过了指定的长度
    if len(line)>num_steps:
        #截断文本序列，取前num_steps个词元
        return line[:num_steps]
    return line+[padding_token]*(num_steps-len(line))




#转换成小批量数据集用于训练
def build_array_nmt(lines,vocab,num_steps):
    """将机器法尼亚的文本序列转换成小批量"""
    lines=[vocab[l] for l in lines]
    lines=[l+[vocab['<eos>']] for l in lines ]
    #构建小批量数据集的张量表示
    array=torch.tensor([truncate_pad(l,num_steps,vocab['<pad>']) for l in lines])
    # 计算原始句子的实际长度
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len  # valid_len 为原始句子的实际长度

#训练模型
def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回翻译数据集的迭代器和词汇表"""
    #预处理原始数据集
    text=preprocess_nmt(read_data_nmt())
    #对预处理后的文本进行词元化
    source,target=tokenize_nmt(text,num_examples)
    # 创建源语言词汇表对象
    src_vocab = d2l.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    # 创建目标语言词汇表对象
    tgt_vocab = d2l.Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    #将源语言文本序列转换为小批量数据集的张量表示和实际长度
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    # 将目标语言文本序列转换为小批量数据集的张量表示和实际长度
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    # 构建数据集的张量表示和实际长度的元组
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    # 加载数据集并创建迭代器
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab

"""以上读取数据部分"""
def train_seq2seq_custom(net, train_iter, lr, num_epochs, tgt_vocab, src_vocab, device):
    """适配自定义Transformer的训练循环"""

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.Embedding:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.NLLLoss()  # 配合Generator的log_softmax输出
    net.train()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[10, num_epochs])

    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # 累计损失和词元数量
        for batch in train_iter:
            src, src_valid_len, tgt, tgt_valid_len = [x.to(device) for x in batch]

            # 1. 生成source_mask（屏蔽源序列Padding）
            batch_size, src_seq_len = src.shape
            source_mask = torch.ones((batch_size, 1, src_seq_len), device=device, dtype=torch.bool)
            source_mask = source_mask.masked_fill(src.unsqueeze(1) == src_vocab['<pad>'], False)

            # 2. 解码器输入和目标标签
            dec_input = tgt[:, :-1]  # 解码器输入：去掉最后一个词
            y = tgt[:, 1:]  # 目标标签：去掉第一个词

            # 3. 生成target_mask（屏蔽目标序列Padding和未来信息）
            dec_seq_len = dec_input.shape[1]
            future_mask = subsequent_mask(dec_seq_len).to(device, dtype=torch.bool)
            padding_mask = torch.ones((batch_size, 1, dec_seq_len), device=device, dtype=torch.bool)
            padding_mask = padding_mask.masked_fill(dec_input.unsqueeze(1) == tgt_vocab['<pad>'], False)
            target_mask = padding_mask & future_mask

            # 4. 模型前向传播
            optimizer.zero_grad()
            y_hat = net(src, dec_input, source_mask, target_mask)

            # 5. 计算损失（忽略Padding）
            y_hat = y_hat.reshape(-1, y_hat.shape[-1])
            y = y.reshape(-1)
            mask = (y != tgt_vocab['<pad>']).float()

            # 额外校验：确保标签在词表范围内（防止CUDA断言错误）
            if device.type == 'cuda':
                vocab_size = len(tgt_vocab)
                if (y >= vocab_size).any() or (y < 0).any():
                    # 修正无效标签（实际不应走到这一步，因数据处理已修正）
                    y = torch.clamp(y, 0, vocab_size - 1)

            loss = criterion(y_hat, y) * mask
            loss = loss.sum() / mask.sum()  # 按有效词元数平均

            # 6. 反向传播与优化
            loss.backward()
            optimizer.step()

            # 7. 累计指标
            metric.add(loss.item() * mask.sum().item(), mask.sum().item())

        # 打印训练日志
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
        print(f'epoch {epoch + 1}, loss {metric[0] / metric[1]:.3f}, '
              f'time {timer.stop():.1f} sec')
    print(f'最终损失 {metric[0] / metric[1]:.3f}, 速度 {metric[1] / timer.sum():.1f} tokens/sec on {device}')

if __name__ == '__main__':
    # 加载数据（英语→法语翻译）
    num_hiddens = 128  # d_model：词嵌入维度
    num_layers = 2  # 编码器/解码器层数
    dropout = 0.1
    batch_size = 32
    num_steps = 10  # 序列最大长度
    lr = 0.001
    num_epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_ff = 512  # 前馈网络中间层维度
    num_heads = 4  # 注意力头数
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps, num_examples=1000)
    net=make_model(
        len(src_vocab),
        len(tgt_vocab),
        N=6,
        d_model=512,
        d_ff=2048,
        head=8,
        dropout=0.1)
train_seq2seq_custom(net, train_iter, lr, num_epochs, tgt_vocab, src_vocab, device)

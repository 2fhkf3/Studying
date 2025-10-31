import collections
import math
import  os
import torch
from d2l import torch as d2l
from torch import nn

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


"""
#绘制每个文本序列所包含的标记数量的直方图，根据句子长度做的直方图
#设置图像的大小
d2l.set_figsize()
#绘制每个文本序列所包含的标记数量的直方图
_,_,patches=d2l.plt.hist(
    [[len(l) for l in source],[len(l) for l in target]],
    label=['source','target']
)
for patch in patches[1].patches:
    patch.set_hatch('/')
d2l.plt.legend(loc='upper right')
"""





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

# 读出 “英语-法语” 数据集中第一个小批量数据
# 加载翻译数据集的迭代器和词汇表，设置每个小批量的大小和序列长度
train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
# 遍历数据迭代器，获取每个小批量的数据和有效长度
for X, X_valid_len, Y, Y_valid_len in train_iter:
    # 打印源语言序列的张量表示（整数类型）
    print('X:', X.type(torch.int32))
    # 打印源语言序列的有效长度
    print('valid lengths for X:', X_valid_len)
    # 打印目标语言序列的张量表示（整数类型）
    print('Y:', Y.type(torch.int32))
    # 打印目标语言序列的有效长度
    print('valid lengths for Y:', Y_valid_len)
    # 跳出循环，只打印第一个小批量数据
    break

#实现循环神经网络编码器
class Seq2SeqEncoder(d2l.Encoder):
    def __init__(self,vocab_size,embed_size,num_hiddens,num_layers,dropout=0,**kwargs):
        super().__init__(**kwargs)
        #创建一个嵌入层，用于将输入的单词索引转换为词嵌入向量
        self.embedding=nn.Embedding(vocab_size,embed_size)
        #创建一个GRU循环神经网络模型
        self.rnn=nn.GRU(embed_size,num_hiddens,num_layers,dropout=dropout)

    def forward(self,x,*args):
        #将输入序列进行词嵌入操作
        x=self.embedding(x)
        #将输入序列的维度进行专职，以适应rnn模型的输入格式要求
        x=x.permute(1,0,2)
        #将转置后的输入序列输入导rnn模型中，得到输出和最终的隐藏状态
        output,state=self.rnn(x)
        return output,state

# 上述编码器的实现
# 创建一个Seq2SeqEncoder对象，设置词汇表大小为10，嵌入维度为8，隐藏状态维度为16，层数为2
encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
# 将编码器设置为评估模式，这将影响一些层的行为，如dropout层
encoder.eval()
# 创建一个形状为(4, 7)的输入张量X，用于模拟4个样本，每个样本有7个单词
X = torch.zeros((4,7), dtype=torch.long)
# 将输入张量X传递给编码器，得到输出张量output和最终隐藏状态state
output, state = encoder(X)
#打印输出张量的形状
print(output.shape)
print(state.shape)

#解码器
class Seq2SeqDecoder(d2l.Decoder):
    """用于序列到序列学习的循环神经网络解码器"""
    def __init__(self,vocab_size,embed_size,num_hiddens,num_layers,dropout=0,**kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        # 创建一个嵌入层，用于将输入的单词索引转换为词嵌入向量
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 创建一个GRU循环神经网络模型
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        #创建一个全连接层，用于将隐藏状态映射到词汇表大小的向量
        self.dense=nn.Linear(num_hiddens,vocab_size)

    def init_state(self,enc_outputs,*args):
        #返回编码器输出的最终隐藏状态作为解码器的初始隐藏状态
        return enc_outputs[1]

    def forward(self,x,state):
        #将输入序列进行词嵌入操作，并进行维度转置
        x=self.embedding(x).permute(1,0,2)
        #将编码器的最终隐藏状态进行复制，用于和每个解码器输入进行拼接
        context=state[-1].repeat(x.shape[0],1,1)
        #将词嵌入序列和编码器最终隐藏状态拼接起来作为解码器输入
        X_and_context = torch.cat((x, context), 2)
        # 将拼接后的输入序列和初始隐藏状态输入到RNN模型中
        output, state = self.rnn(X_and_context, state)
        # 将RNN模型的输出通过全连接层映射到词汇表大小的向量，并进行维度转置
        output = self.dense(output).permute(1, 0, 2)
        # 返回输出和最终隐藏状态
        return output, state

# 实例化解码器
# 创建一个Seq2SeqDecoder对象，设置词汇表大小为10，嵌入维度为8，隐藏状态维度为16，层数为2
decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
# 将解码器设置为评估模式，这将影响一些层的行为，如dropout层
decoder.eval()
# 使用编码器的输出来初始化解码器的隐藏状态
state = decoder.init_state(encoder(X))
# 将输入张量X和初始化的隐藏状态传递给解码器，得到输出张量output和更新后的隐藏状态state
output, state = decoder(X, state)
# 打印输出张量和隐藏状态的形状
print(output.shape, state.shape)


#通过零值化屏蔽不相关的项
def sequence_mask(x,valid_len,value=0):
    """在序列中屏蔽不相关的项"""
    #获取序列的最大长度
    maxlen = x.size(1)
    # 创建一个掩码，标记不相关的项为False
    mask=torch.arange((maxlen),dtype=torch.float32,device=x.device)[None,:]<valid_len[:,None]
    #将不相干的项零值化
    x[~mask]=value
    return x
# 创建一个输入张量X，用于演示
X = torch.tensor([[1,2,3],[4,5,6]])
# 调用sequence_mask函数，对输入张量X进行屏蔽操作，将填充的项标出来
print(sequence_mask(X, torch.tensor([1,2])))


# 我们还可以使用此函数屏蔽最后几个轴上的所有项
# 创建一个全为1的输入张量X，用于演示
X = torch.ones(2,3,4)
# 调用sequence_mask函数，对输入张量X进行屏蔽操作，将最后几个轴上的所有项标出来，使用-1进行填充
print(sequence_mask(X, torch.tensor([1,2]),value=-1))


# 通过扩展softmax交叉熵损失函数来遮蔽不相关的预测
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""
    def forward(self,pred,label,valid_len):
        #创建一个与标签张量label星湖在那个相同的张量，所有元素都为1，用作权重
        weights=torch.ones_like(label)
        # 使用sequence_mask函数对权重张量进行遮蔽操作，将不相关的项标出来
        weights = sequence_mask(weights, valid_len)
        # 设置损失函数的计算方式为不进行降维
        self.reduction = 'none'
        # 调用父类的forward方法计算未加权的交叉熵损失
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0, 2, 1), label)
        # 将未加权的损失乘以权重，然后在第1个维度上求均值，得到加权的损失
        weighted_loss = (unweighted_loss * weights).mean(dim=1)  # 有效的留下来，没效的全部变为0
        return  weighted_loss

# 代码健壮性检查
# 实例化MaskedSoftmaxCELoss类，创建一个损失函数对象
loss = MaskedSoftmaxCELoss()
# 调用损失函数对象的forward方法，计算损失
loss(torch.ones(3,4,10), torch.ones((3,4),dtype=torch.long),torch.tensor([4,2,0]))

#训练
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """训练序列到序列模型"""
    def xavier_init_weights(m):
        # 如果是线性层
        if type(m) == nn.Linear:
            # 使用Xavier均匀初始化权重
            nn.init.xavier_uniform_(m.weight)
        # 如果是GRU层
        if type(m) == nn.GRU:
            # 对于GRU层的每个参数
            for param in m._flat_weights_names:
                # 如果是权重参数
                if "weight" in param:
                    # 使用Xavier均匀初始化该权重参数
                    nn.init.xavier_uniform_(m._parameters[param])
    # 应用xavier_init_weights函数，对网络模型的参数进行初始化
    net.apply(xavier_init_weights)
    # 将网络模型移动到指定设备上
    net.to(device)
    # 创建Adam优化器，将网络模型的参数传入优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # 创建MaskedSoftmaxCELoss损失函数对象
    loss = MaskedSoftmaxCELoss()
    # 将网络模型设置为训练模式
    net.train()
    # 创建动画绘制对象，用于绘制损失随训练epoch的变化情况
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        # 创建计时器对象，用于计算每个epoch的训练时间
        timer = d2l.Timer()
        # 创建累加器对象，用于累加损失和标记的数量
        metric = d2l.Accumulator(2)
        for batch in data_iter:
            # 将输入数据移动到指定设备上
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            #构造解码器的输入，将bos和去除组后一列的标签张量Y拼接起来
            dec_input = torch.cat([bos, Y[:, :-1]], 1)
            # 前向传播，得到预测结果Y_hat
            Y_hat= net(X, dec_input, X_valid_len)
            # 计算损失
            l = loss(Y_hat, Y, Y_valid_len)
            # 反向传播，计算梯度
            l.sum().backward()
            # 对梯度进行裁剪，防止梯度爆炸
            d2l.grad_clipping(net, 1)
            # 计算标记的数量
            num_tokens = Y_valid_len.sum()
            # 更新模型参数
            optimizer.step()
            # 使用torch.no_grad()上下文管理器，关闭梯度计算，避免计算图的构建
            with torch.no_grad():
                # 累加损失和标记的数量
                metric.add(l.sum(), num_tokens)
            # 每10个epoch打印一次损失
        if (epoch + 1) % 10 == 0:
            # 绘制损失随训练epoch的变化情况
            animator.add(epoch + 1, (metric[0] / metric[1],))
            # 打印最终的损失值和训练速度
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
              f'tokens/sec on {str(device)}')


# 创建和训练一个循环神经网络 “编码器-解码器” 模型
# 设置嵌入层大小(embed_size)、隐藏层大小(num_hiddens)、层数(num_layers)和丢弃率(dropout)的数值
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
# 设置批量大小(batch_size)和时间步数(num_steps)的数值
batch_size, num_steps = 64, 10
# 设置学习率(lr)、训练轮数(num_epochs)和设备(device)
lr, num_epochs, device = 0.005, 300, d2l.try_gpu()
# 加载训练数据集和词汇表
train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)
# 创建编码器和解码器

encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
# 创建整个编码器-解码器模型
net = d2l.EncoderDecoder(encoder, decoder)
train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)



#预测
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                   device, save_attention_weights=False):
    """序列到序列模型的预测"""
    #将模型设为评估模式，用于预测
    net.eval()
    #将源语言句子转换为词元，并添加<eos>作为结束符
    src_tokens=src_vocab[src_sentence.lower().split(' ')]+[src_vocab['<eoc>']]
    #计算有效长度
    enc_valid_len=torch.tensor([len(src_tokens)],device=device)
    # 截断或填充源语言句子
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 将源语言句子转换为张量，并添加一个维度表示批量大小
    enc_X = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    # 使用编码器生成编码器输出
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    # 初始化解码器状态
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 初始化解码器输入
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    # 初始化输出序列和注意力权重序列
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        # 使用解码器生成输出和更新解码器状态
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 获取解码器输出中概率最高的词元作为下一个解码器输入
        dec_X = Y.argmax(dim=2)
        # 获取预测结果
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 如果需要保存注意力权重，则将当前的注意力权重添加到注意力权重序列中
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 判断是否预测到结束符
        if pred == tgt_vocab['<eos>']:
            break
        # 将预测结果添加到输出序列
        output_seq.append(pred)
    # 将输出序列转换为字符串，并返回输出序列和注意力权重序列（如果需要）
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq

#BLEU的代码实现
def bleu(pred_seq, label_seq, k):
    """计算 BLEU"""
    # 将预测序列和标签序列分割成词元列表
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    # 计算预测序列和标签序列的长度
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    # 初始化得分，并根据预测序列和标签序列的长度比例进行调整
    score = math.exp(min(0, 1 - len_label / len_pred))
    # 对每个n-gram进行计算，其中k为最大n-gram的大小
    for n in range(1, k + 1):
        # 初始化匹配次数和标签序列中的n-gram计数器
        num_matches, label_subs = 0, collections.defaultdict(int)
        # 遍历标签序列，计算标签序列中的n-gram出现次数
        for i in range(len_label - n + 1):
            # 更新标签序列中n-gram的计数
            label_subs[''.join(label_tokens[i:i + n])] += 1
        # 遍历预测序列，统计预测序列中与标签序列n-gram匹配的次数
        for i in range(len_pred - n + 1):
            # 如果预测序列中的n-gram在标签序列中出现，则增加匹配次数，并减少标签序列中该n-gram的计数
            if label_subs[''.join(pred_tokens[i:i + n])] > 0:
                num_matches += 1
                label_subs[''.join(pred_tokens[i:i + n])] -= 1
        # 根据匹配次数和预测序列的长度计算得分
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    # 返回计算得到的BLEU得分
    return score

# 将几个英语句子翻译成法语
# 定义英语句子列表
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
# 定义法语句子列表
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
# 使用zip函数迭代英语句子和法语句子的对应元素
for eng, fra in zip(engs, fras):
    # 调用predict_seq2seq函数进行翻译预测，并获取翻译结果和注意力权重序列
    translation, attention_weight_seq = predict_seq2seq(net, eng, src_vocab, tgt_vocab, num_steps, device)
    # 调用bleu函数计算翻译结果的BLEU分数
    # 打印英语句子、翻译结果和BLEU分数
    print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')
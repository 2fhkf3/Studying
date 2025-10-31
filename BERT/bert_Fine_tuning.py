import json
import os
import re
import torch
from torch import nn
from d2l import torch as d2l



#定义函数，用于处理文本数据：读取和预处理 SNLI 数据集
def read_snli(data_dir,is_train):
    #定义函数，用于处理文本数据
    def extract_text(s):
        #去除左括号
        s=re.sub('\\(','',s)
        #去除右括号
        s=re.sub('\\)','',s)
        #去除多余的空格
        s=re.sub('\\s{2,}','',s)
        return s.strip()
    #标签映射表
    label_set={'entailment':0,'contradiction':1,'neutral':2}
    #根据是否是训练集选择相应的文件名
    file_name = os.path.join(data_dir, 'snli_1.0_train.txt'
    if is_train else 'snli_1.0_test.txt')
    #打开文件并逐行读取数据
    with open(file_name,'r') as f:
        rows=[row.split('\t') for row in f.readlines()[1:]]
    #提取前提、假设和标签
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels

class SNLIDataset(torch.utils.data.Dataset):
    #初始化方法
    def __init__(self,dataset,num_steps,vocab=None):
        self.num_steps=num_steps
        #分词得到前提的标记列表
        all_premise_tokens=d2l.tokenize(dataset[0])
        # 分词得到假设的标记列表
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
          #构建词汇表
           self.vocab=d2l.Vocab(all_premise_tokens + all_hypothesis_tokens,
                                   min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab=vocab
        #填充后的前提标记张量
        self.premises=self._pad(all_premise_tokens)
        # 填充后的假设标记张量
        self.hypotheses = self._pad(all_hypothesis_tokens)
        # 标签张量
        self.labels = torch.tensor(dataset[2])
        print('read ' + str(len(self.premises)) + ' examles')

    def _pad(self, lines):
        # 辅助方法，对标记列表进行填充
        return torch.tensor([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
            for line in lines])

    def __getitem__(self, idx):
        # 获取数据项的方法
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        # 返回数据集的长度
        return len(self.premises)

def load_data_snli(batch_size,num_steps=50):
    num_workers=0
    data_dir = './data/snli_1.0/snli_1.0'
    #读取训练集数据
    train_data=read_snli(data_dir,True)
    #读取测试集数据
    test_data=read_snli(data_dir,False)
    #创建训练集数据集对象
    train_set = SNLIDataset(train_data, num_steps)
    # 创建测试集数据集对象，并共享训练集的词汇表
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    # 创建训练集数据迭代器
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                             shuffle=True, num_workers=num_workers)
    # 创建测试集数据迭代器
    test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                            shuffle=False, num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab



#微调BERT
import os
import torch
from d2l import torch as d2l
# Loading Pretrained BERT
# 定义预训练的BERT模型的数据源链接和哈希值
d2l.DATA_HUB['bert.base'] = (d2l.DATA_URL + 'bert.base.torch.zip',
                             '225d66f04cae318b841a13d32af3acc165f253ac')
d2l.DATA_HUB['bert.small'] = (d2l.DATA_URL + 'bert.small.torch.zip',
                              'c72329e68a732bef0452e4b96a1c341c8910f81f')
#加载预训练的 BERT 模型及其词汇表
def load_pretrained_model(pretrained_model, num_hiddens, ffn_num_hiddens,
                          num_heads, num_layers, dropout, max_len, devices):
    # 下载和提取预训练模型的数据文件
    data_dir = d2l.download_extract(pretrained_model)
    # 创建词汇表对象，并加载词汇表索引和标记的映射关系
    vocab = d2l.Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir, 'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(vocab.idx_to_token)}
# 创建BERT模型对象
    bert = d2l.BERTModel(len(vocab), num_hiddens, ffn_num_hiddens=ffn_num_hiddens,num_heads=4,num_blks=2,dropout=0.2,max_len=max_len)
    # 加载预训练的BERT参数
    bert.load_state_dict(torch.load(os.path.join(data_dir, 'pretrained.params'),weights_only=True))
    return bert, vocab

devices = d2l.try_all_gpus()
# 加载预训练的BERT模型和词汇表
bert, vocab = load_pretrained_model(
    'bert.small', num_hiddens=256, ffn_num_hiddens=512, num_heads=4,
    num_layers=2, dropout=0.1, max_len=512, devices=devices)


# The Dataset for Fine-Tuning BERT
class SNLIBERTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, max_len, vocab=None):
        # 将前提和假设的文本转换为词元序列，并存储在all_premise_hypothesis_tokens中
        all_premise_hypothesis_tokens = [[
            p_tokens, h_tokens] for p_tokens, h_tokens in zip(
            *[d2l.tokenize([s.lower() for s in sentences])
              for sentences in dataset[:2]])]
        # 存储标签
        self.labels = torch.tensor(dataset[2])
        self.vocab = vocab
        self.max_len = max_len
        # 对所有前提和假设的词元序列进行预处理
        (self.all_token_ids, self.all_segments,
         self.valid_lens) = self._preprocess(all_premise_hypothesis_tokens)
        print('read ' + str(len(self.all_token_ids)) + ' examples')

    def _preprocess(self, all_premise_hypothesis_tokens):
        # 多进程处理词元序列
        out = map(self._mp_worker, all_premise_hypothesis_tokens)
        out = list(out)
        # 提取预处理后的结果
        all_token_ids = [token_ids for token_ids, segments, valid_len in out]
        all_segments = [segments for token_ids, segments, valid_len in out]
        valid_lens = [valid_len for token_ids, segments, valid_len in out]
        return (torch.tensor(all_token_ids, dtype=torch.long),
                torch.tensor(all_segments, dtype=torch.long),
                torch.tensor(valid_lens))

    def _mp_worker(self, premise_hypothesis_tokens):
        # 处理单个前提和假设的词元序列
        p_tokens, h_tokens = premise_hypothesis_tokens
        # 截断前提和假设的词元序列
        self._truncate_pair_of_tokens(p_tokens, h_tokens)
        # 获取词元和片段标记
        tokens, segments = d2l.get_tokens_and_segments(p_tokens, h_tokens)
        # 将词元序列转换为索引序列，补齐至最大长度
        token_ids = self.vocab[tokens] + [self.vocab['<pad>']] * (self.max_len - len(tokens))
        # 补齐片段标记至最大长度
        segments = segments + [0] * (self.max_len - len(segments))
        # 记录有效长度
        valid_len = len(tokens)
        return token_ids, segments, valid_len

    def _truncate_pair_of_tokens(self, p_tokens, h_tokens):
        # 不断截断前提和假设的词元序列，使总长度不超过最大长度减3
        while len(p_tokens) + len(h_tokens) > self.max_len - 3:
            if len(p_tokens) > len(h_tokens):
                p_tokens.pop()
            else:
                h_tokens.pop()

    def __getitem__(self, idx):
        # 返回数据样本和标签
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx]), self.labels[idx]

    def __len__(self):
        # 返回数据集的样本数量
        return len(self.all_token_ids)
# Generate training and testing examples
# 生成训练和测试样本
batch_size, max_len, num_workers = 32, 128, 0
data_dir = '.\data\snli_1.0\snli_1.0'
# 创建训练集和测试集的 SNLIBERTDataset 对象
train_set = SNLIBERTDataset(d2l.read_snli(data_dir, True), max_len, vocab)
test_set = SNLIBERTDataset(d2l.read_snli(data_dir, False), max_len, vocab)
# 创建训练集和测试集的数据迭代器
train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers = num_workers)
test_iter = torch.utils.data.DataLoader(test_set, batch_size, num_workers = num_workers)


# This MLP transforms the BERT representation of the special "<cls>" token into three outputs of natural language inference
# 定义一个 MLP，将特殊的 "<cls>" 标记的 BERT 表示转换为三个自然语言推断的输出
class BERTClassifier(nn.Module):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        # BERT 编码器
        self.encoder = bert.encoder
        # BERT 隐藏层
        self.hidden = bert.hidden
        # 输出层，将隐藏层的输出映射到三个类别
        self.output = nn.Linear(256, 3)

    def forward(self, inputs):
        # 输入数据
        tokens_X, segments_X, valid_lens_x = inputs
        # BERT 编码器的输出
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        # 将特殊标记的表示进行线性变换得到最终输出
        return self.output(self.hidden(encoded_X[:, 0, :]))


# 创建 BERTClassifier 对象
net = BERTClassifier(bert)

# The training
# 训练过程
# 学习率和训练轮数
lr, num_epochs = 1e-4, 5
# Adam 优化器
trainer = torch.optim.Adam(net.parameters(), lr=lr)
# 交叉熵损失函数，不进行降维
loss = nn.CrossEntropyLoss(reduction='none')
# 调用 d2l.train_ch13 函数进行模型的训练
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
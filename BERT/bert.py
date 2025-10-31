from torch import nn
import os
import random
import torch
from d2l import torch as d2l

# -------------------------- 1. 工具函数：生成token和段标记 --------------------------
def get_tokens_and_segments(tokens_a, tokens_b=None):
    tokens = ['<CLS>'] + tokens_a + ['<SEP>']
    segments = [0] * (len(tokens_a) + 2)  # <CLS>和<SEP>属于句子A
    if tokens_b is not None:
        tokens += tokens_b + ['<SEP>']
        segments += [1] * (len(tokens_b) + 1)  # 句子B的段标记为1
    return tokens, segments

# -------------------------- 2. BERT编码器 --------------------------
class BERTEncoder(nn.Module):
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, max_len=1000, key_size=768, query_size=768, value_size=768,** kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)  # token嵌入
        self.segment_embedding = nn.Embedding(2, num_hiddens)  # 段嵌入（0/1）
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))  # 位置嵌入（修正：移到循环外）
        # 堆叠Transformer编码器块
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                f"{i}",
                d2l.TransformerEncoderBlock(
                    num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias=True
                )
            )

    def forward(self, tokens, segments, valid_lens):
        # token嵌入 + 段嵌入 + 位置嵌入
        x = self.token_embedding(tokens) + self.segment_embedding(segments)
        x += self.pos_embedding.data[:, :x.shape[1], :]  # 位置嵌入截断到序列长度
        # 经过所有Transformer块
        for blk in self.blks:
            x = blk(x, valid_lens)
        return x

# -------------------------- 3. MLM（掩码语言模型） --------------------------
class MaskLM(nn.Module):
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(num_inputs, num_hiddens),
            nn.ReLU(),
            nn.LayerNorm(num_hiddens),
            nn.Linear(num_hiddens, vocab_size)  # 输出词汇表大小的概率
        )

    def forward(self, x, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)  # 展平为一维（batch_size * num_pred）
        batch_size = x.shape[0]
        # 生成批量索引（每个样本对应num_pred个位置）
        batch_idx = torch.arange(batch_size, device=x.device).repeat_interleave(num_pred_positions)
        # 提取掩码位置的特征
        masked_X = x[batch_idx, pred_positions]
        masked_X = masked_X.reshape(batch_size, num_pred_positions, -1)  # 恢复批量维度
        # 预测掩码位置的token
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat

# -------------------------- 4. NSP（下一句预测） --------------------------
class NextSentencePred(nn.Module):
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)  # 二分类：是/否下一句

    def forward(self, x):
        return self.output(x)

# -------------------------- 5. 完整BERT模型 --------------------------
class BERTModel(nn.Module):
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, max_len=1000, key_size=768, mlm_in_features=768, nsp_in_features=768):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(
            vocab_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
            num_heads, num_layers, dropout, max_len, key_size
        )
        self.hidden = nn.Sequential(nn.Linear(num_hiddens, num_hiddens), nn.Tanh())  # NSP的隐藏层
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)  # MLM头部
        self.nsp = NextSentencePred(nsp_in_features)  # NSP头部

    def forward(self, tokens, segments, valid_lens=None, pred_position=None):
        encoded_x = self.encoder(tokens, segments, valid_lens)
        # MLM预测（需传入掩码位置）
        mlm_Y_hat = self.mlm(encoded_x, pred_position) if pred_position is not None else None
        # NSP预测（用<CLS>位置的特征）
        nsp_Y_hat = self.nsp(self.hidden(encoded_x[:, 0, :]))
        return encoded_x, mlm_Y_hat, nsp_Y_hat

# -------------------------- 6. WikiText-2数据集处理 --------------------------
def _read_wiki(data_dir):
    """读取WikiText-2训练集，返回段落列表（每个段落是句子列表）"""
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 按“ . ”分割句子，过滤掉句子数<2的段落，转为小写
    paragraphs = [line.strip().lower().split(' . ') for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)  # 打乱段落
    return paragraphs

def _get_next_sentence(sentence, next_sentence, paragraphs):
    """生成NSP的句子对（50%连续，50%随机）"""
    if random.random() < 0.5:
        is_next = True
    else:
        # 随机选一个段落，再随机选一个句子作为next_sentence
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next

def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    """从单个段落生成NSP样本"""
    nsp_samples = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = _get_next_sentence(paragraph[i], paragraph[i+1], paragraphs)
        # 检查长度：<CLS> + A + <SEP> + B + <SEP> 总长度不超过max_len
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        # 生成token和段标记
        tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
        nsp_samples.append((tokens, segments, is_next))
    return nsp_samples

def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds, vocab):
    """替换token为<mask>、原token或随机token（MLM规则）"""
    mlm_input_tokens = tokens.copy()  # 复制原始token列表
    pred_positions_and_labels = []
    random.shuffle(candidate_pred_positions)  # 随机选掩码位置

    for pos in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break  # 达到掩码数量上限
        # MLM规则：80%<mask>，10%原token，10%随机token
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            if random.random() < 0.5:
                masked_token = tokens[pos]  # 原token
            else:
                # 随机token：从词汇表中选（修正：索引转token）
                masked_token = vocab.idx_to_token[random.randint(0, len(vocab) - 1)]
        mlm_input_tokens[pos] = masked_token
        pred_positions_and_labels.append((pos, tokens[pos]))  # 记录原始token（作为标签）
    return mlm_input_tokens, pred_positions_and_labels

def _get_mlm_data_from_tokens(tokens, vocab):
    """从token列表生成MLM样本（输入、掩码位置、标签）"""
    # 筛选候选掩码位置（排除<CLS>和<SEP>）
    candidate_pred_positions = [i for i, token in enumerate(tokens) if token not in ['<CLS>', '<SEP>']]
    # 掩码数量：序列长度的15%（至少1个）
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    # 生成掩码输入和标签
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab
    )
    # 按位置排序（确保顺序与序列一致）
    pred_positions_and_labels.sort(key=lambda x: x[0])
    pred_positions = [pos for pos, _ in pred_positions_and_labels]
    mlm_labels = [token for _, token in pred_positions_and_labels]
    # 转为词汇表索引
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_labels]

def _pad_bert_inputs(examples, max_len, vocab):
    """填充所有样本到max_len，转换为张量"""
    max_num_mlm_preds = round(max_len * 0.15)  # 最大掩码数量
    # 初始化存储列表
    all_token_ids, all_segments, valid_lens = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []

    for (token_ids, pred_positions, mlm_label_ids, segments, is_next) in examples:
        # 1. 填充token_ids（不足用<pad>）
        pad_len = max_len - len(token_ids)
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * pad_len, dtype=torch.long))
        # 2. 填充segments（不足用0）
        all_segments.append(torch.tensor(segments + [0] * pad_len, dtype=torch.long))
        # 3. 有效长度（原始token数量）
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        # 4. 填充掩码位置（不足用0）
        pad_pred_len = max_num_mlm_preds - len(pred_positions)
        all_pred_positions.append(torch.tensor(pred_positions + [0] * pad_pred_len, dtype=torch.long))
        # 5. MLM权重（有效位置1.0，填充0.0）
        all_mlm_weights.append(torch.tensor([1.0]*len(mlm_label_ids) + [0.0]*pad_pred_len, dtype=torch.float32))
        # 6. 填充MLM标签（不足用0）
        all_mlm_labels.append(torch.tensor(mlm_label_ids + [0] * pad_pred_len, dtype=torch.long))
        # 7. NSP标签（True→1，False→0）
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))

    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)

class _WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, paragraphs, max_len):
        # 1. 段落分词（每个句子转为token列表）
        paragraphs = [d2l.tokenize(p, token='word') for p in paragraphs]
        # 2. 提取所有句子，构建词汇表
        sentences = [s for p in paragraphs for s in p]
        self.vocab = d2l.Vocab(
            sentences, min_freq=5, reserved_tokens=['<pad>', '<mask>', '<CLS>', '<SEP>']
        )
        # 3. 收集所有NSP样本
        nsp_examples = []
        for p in paragraphs:
            nsp_examples.extend(_get_nsp_data_from_paragraph(p, paragraphs, self.vocab, max_len))
        # 4. 为NSP样本添加MLM标注
        mlm_examples = []
        for tokens, segments, is_next in nsp_examples:
            mlm_token_ids, pred_positions, mlm_label_ids = _get_mlm_data_from_tokens(tokens, self.vocab)
            mlm_examples.append((mlm_token_ids, pred_positions, mlm_label_ids, segments, is_next))
        # 5. 填充并转换为张量
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(mlm_examples, max_len, self.vocab)

    def __getitem__(self, idx):
        """返回单条样本（7个张量）"""
        return (self.all_token_ids[idx], self.all_segments[idx], self.valid_lens[idx],
                self.all_pred_positions[idx], self.all_mlm_weights[idx],
                self.all_mlm_labels[idx], self.nsp_labels[idx])

    def __len__(self):
        """样本总数"""
        return len(self.all_token_ids)

def load_data_wiki(batch_size, max_len):
    """加载WikiText-2数据集，返回DataLoader和词汇表"""
    data_dir = 'G:/pythonProject6/db1ec-main/db1ec-main/wikitext-2-v1/wikitext-2'  # 正斜杠避免转义问题
    paragraphs = _read_wiki(data_dir)
    dataset = _WikiTextDataset(paragraphs, max_len)
    # Windows下num_workers设为0，避免多进程错误
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, num_workers=0)
    return train_iter, dataset.vocab

# -------------------------- 7. 训练函数 --------------------------
def _get_batch_loss_bert(net, loss_fn, vocab_size, tokens_X, segments_X, valid_lens_X,
                         pred_positions_X, mlm_weights_X, mlm_Y, nsp_Y):
    """计算单批数据的MLM和NSP损失"""
    # 前向传播
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X, valid_lens_X, pred_positions_X)
    # MLM损失：只计算有效掩码位置的损失（乘mlm_weights_X）
    mlm_loss = loss_fn(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) * mlm_weights_X.reshape(-1, 1)
    mlm_loss = mlm_loss.sum() / (mlm_weights_X.sum() + 1e-8)  # 除以有效掩码数量
    # NSP损失
    nsp_loss = loss_fn(nsp_Y_hat, nsp_Y)
    # 总损失
    total_loss = mlm_loss + nsp_loss
    return mlm_loss, nsp_loss, total_loss

def train_bert(train_iter, net, loss_fn, vocab_size, device, num_steps):
    """训练BERT模型"""
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    step = 0
    timer = d2l.Timer()
    animator = d2l.Animator(xlabel='step', ylabel='loss', xlim=[1, num_steps], legend=['MLM', 'NSP'])
    metric = d2l.Accumulator(4)  # 存储：mlm_loss_sum, nsp_loss_sum, sample_count, step_count

    while step < num_steps:
        for batch in train_iter:
            # 数据移到设备
            tokens_X, segments_X, valid_lens_X, pred_positions_X, mlm_weights_X, mlm_Y, nsp_Y = [
                x.to(device) for x in batch
            ]
            # 梯度清零
            optimizer.zero_grad()
            # 计算损失
            mlm_loss, nsp_loss, total_loss = _get_batch_loss_bert(
                net, loss_fn, vocab_size, tokens_X, segments_X, valid_lens_X,
                pred_positions_X, mlm_weights_X, mlm_Y, nsp_Y
            )
            # 反向传播+更新
            total_loss.backward()
            optimizer.step()
            # 记录指标
            metric.add(mlm_loss.item(), nsp_loss.item(), tokens_X.shape[0], 1)
            # 绘图
            animator.add(step + 1, (metric[0]/metric[3], metric[1]/metric[3]))
            step += 1
            if step == num_steps:
                break

    # 打印训练结果
    print(f'MLM平均损失: {metric[0]/metric[3]:.3f}')
    print(f'NSP平均损失: {metric[1]/metric[3]:.3f}')
    print(f'训练速度: {metric[2]/timer.stop():.1f} 样本对/秒 (设备: {device})')

"""训练模式"""
# 超参数
batch_size = 4  # 原为512，小批量更适合调试
max_len = 64
num_steps = 50  # 预训练步数
# 加载数据
train_iter, vocab = load_data_wiki(batch_size, max_len)
print(f'词汇表大小: {len(vocab)}')
# 初始化模型
net = BERTModel(
        vocab_size=len(vocab),
        num_hiddens=128,
        norm_shape=[128],
        ffn_num_input=128,
        ffn_num_hiddens=256,
        num_heads=2,
        num_layers=2,
        dropout=0.2,
        key_size=128,
        mlm_in_features=128,
        nsp_in_features=128
    )
# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'使用设备: {device}')
# 损失函数
loss_fn = nn.CrossEntropyLoss()
# 开始预训练
train_bert(train_iter, net, loss_fn, len(vocab), device, num_steps)
####################
#使用BERT表示文本
def get_bert_encoding(net,tokens_a,tokens_b=None):
    #获取tokens和segments
    tokens,segments=get_tokens_and_segments(tokens_a,tokens_b)
    #将tokens转换为token_ids，并添加批次维度
    token_ids=torch.tensor(vocab[tokens],device=device).unsqueeze(0)
    #将segments转换为tensor，并添加批次维度
    segments=torch.tensor(segments,device=device).unsqueeze(0)
    #计算有效长度并添加批次维度
    valid_len=torch.tensor(len(tokens),device=device).unsqueeze(0)
    #进行bert进行编码
    encoded_x,_,_=net(token_ids,segments,valid_len)
    return encoded_x

tokens_a=['a','crane','is','flying']
#使用get_bert_encoding函数对句子进行编码
encoded_text=get_bert_encoding(net,tokens_a)
#提取编码后的句子的CLS标记
encoded_text_cls=encoded_text[:,0,:]
#提取编码后的句子中”crane"的表示
encoded_text_crane=encoded_text[:,2,:]
#输出编码后的句子的形状，CLS标记的形状以及“crane"的前三个表示值
print(encoded_text.shape,encoded_text_cls.shape,encoded_text_crane[0][:3])

#现在考虑句子对"a crane driver came" 和 "he just left"
tokens_a,tokens_b=['a', 'crane', 'driver', 'came'], ['he', 'just', 'left']
#使用get_bert_encoding函数对句子对进行编码
encoded_pair = get_bert_encoding(net, tokens_a, tokens_b)
# 提取编码后的句子对的 CLS 标记
encoded_pair_cls = encoded_pair[:, 0, :]
# 提取编码后的句子对中 "crane" 的表示
encoded_pair_crane = encoded_pair[:, 2, :]
# 输出编码后的句子对的形状、CLS 标记的形状以及 "crane" 的前三个表示值
print(encoded_pair.shape, encoded_pair_cls.shape, encoded_pair_crane[0][:3])
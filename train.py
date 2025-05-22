import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import math
from models import MiniGPT
import json
from torch.utils.data import TensorDataset, DataLoader
from utils.encode import build_vocab, text_to_indices
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

# ================== 数据准备 ==================
# 训练语料
corpus = [w.strip() for w in open("data/corpus.txt",'r',encoding="utf8").readlines()]


vocab, word2idx = build_vocab(corpus)
vocab_size = len(vocab)
print(f"词表大小: {vocab_size}")

# ================== 超参数配置 ==================
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
window_size = config['window_size']

# 创建训练数据（输入和目标）
input_seqs = []
target_seqs = []
for text in corpus:
    indices = text_to_indices(text, word2idx)
    # 补齐到最小长度
    if len(indices) < window_size + 1: 
        padding = [word2idx['<pad>']] * (window_size + 1 - len(indices))
        indices += padding
    # 滑动窗口
    for i in range(0, len(indices) - window_size):
        input_seqs.append(indices[i:i+window_size])
        target_seqs.append(indices[i+1:i+1+window_size])

# ================== 数据集与DataLoader封装 ==================
input_tensor = [torch.tensor(seq, dtype=torch.long) for seq in input_seqs]
target_tensor = [torch.tensor(seq, dtype=torch.long) for seq in target_seqs]

input_tensor = pad_sequence(input_tensor, batch_first=True, padding_value=word2idx['<pad>'])
target_tensor = pad_sequence(target_tensor, batch_first=True, padding_value=word2idx['<pad>'])
dataset = TensorDataset(input_tensor, target_tensor)
dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

# (batch, seq_len)
for batch_inputs, batch_targets in dataloader:
    print("batch_inputs:", batch_inputs)
    print("batch_inputs shape:", batch_inputs.shape)
    print("batch_targets shape:", batch_targets.shape)
    break


# ================== 训练设置 ==================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MiniGPT(vocab_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=config['lr'])
criterion = nn.CrossEntropyLoss()

# ================== 训练循环 ==================
def train():
    model.train()
    for epoch in tqdm(range(config['epochs'])):
        total_loss = 0
        for batch_inputs, batch_targets in dataloader:
            # batch_inputs: (batch, seq_len)
            # batch_targets: (batch, seq_len)
            # 转置为 (seq_len, batch)
            inputs = batch_inputs.transpose(0, 1).to(device)  # (seq_len, batch)
            targets = batch_targets.transpose(0, 1).to(device)  # (seq_len, batch)

            optimizer.zero_grad() 
            output = model(inputs)  
            # output: (seq_len, batch, vocab_size)
            # targets: (seq_len, batch)
            # 交叉熵损失函数要求输入为 (batch, num_classes)，所以需要将 output 和 targets 展平
            output = output.reshape(-1, vocab_size)
            targets = targets.reshape(-1)
            # print(output.shape)
            # print(targets.shape)
            loss = criterion(output, targets)
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新参数
            total_loss += loss.item()  # 累加loss
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}') 
    # 保存参数
    torch.save(model.state_dict(), 'ckpt/minigpt.pt')
    print("模型已保存到 ckpt/minigpt.pt")

train()


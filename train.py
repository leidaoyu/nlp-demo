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
    # 滑动窗口生成定长样本
    for i in range(0, len(indices) - window_size):
        input_seqs.append(indices[i:i+window_size])
        target_seqs.append(indices[i+1:i+1+window_size])

print(input_seqs)
print(target_seqs)

# ================== 训练设置 ==================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MiniGPT(vocab_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=config['lr'])
criterion = nn.CrossEntropyLoss()

# ================== 数据集与DataLoader封装 ==================
input_tensor = [torch.tensor(seq, dtype=torch.long) for seq in input_seqs]
target_tensor = [torch.tensor(seq, dtype=torch.long) for seq in target_seqs]

# Padding

input_tensor = pad_sequence(input_tensor, batch_first=True, padding_value=word2idx['<pad>'])
target_tensor = pad_sequence(target_tensor, batch_first=True, padding_value=word2idx['<pad>'])

dataset = TensorDataset(input_tensor, target_tensor)
dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

# ================== 训练循环 ==================
def train():
    model.train()
    for epoch in tqdm(range(config['epochs'])):
        total_loss = 0
        for batch_inputs, batch_targets in dataloader:
            # 转置为(seq_len, batch)
            inputs = batch_inputs.transpose(0, 1).to(device)
            print(inputs.shape)
            print(inputs)
            targets = batch_targets.transpose(0, 1).to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}')

train()


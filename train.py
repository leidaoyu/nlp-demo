import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import math
from models import MiniGPT

# ================== 数据准备 ==================
# 训练语料
corpus = []

# 构建词表
def build_vocab(corpus):
    words = []
    for text in corpus:
        words.extend(text.split())  # 简单按空格分词（实际应用需要更好分词）
    vocab = list(set(words))
    vocab = ['<pad>', '<bos>', '<eos>'] + vocab  # 添加特殊token
    word2idx = {w:i for i,w in enumerate(vocab)}
    return vocab, word2idx

vocab, word2idx = build_vocab(corpus)
vocab_size = len(vocab)
print(f"词表大小: {vocab_size}")

# 数据预处理（转换为索引）
def text_to_indices(text):
    return [word2idx['<bos>']] + [word2idx[w] for w in text.split()] + [word2idx['<eos>']]

# 创建训练数据（输入和目标）
input_seqs = []
target_seqs = []
for text in corpus:
    indices = text_to_indices(text)
    input_seqs.append(indices[:-1])
    target_seqs.append(indices[1:])



# ================== 训练设置 ==================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MiniGPT(vocab_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# ================== 训练循环 ==================
def train():
    model.train()
    for epoch in range(10):
        total_loss = 0
        for inputs, targets in zip(input_seqs, target_seqs):
            inputs = torch.LongTensor(inputs).unsqueeze(1).to(device)  # (seq_len, 1)
            targets = torch.LongTensor(targets).to(device)
            
            optimizer.zero_grad()
            output = model(inputs)
            
            # 输出形状检查
            print(f"输出形状: {output.shape}")  # (seq_len, batch_size, vocab_size)
            
            loss = criterion(output.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(input_seqs):.4f}')

train()

# ================== 文本生成 ==================
def generate(prompt, max_len=20, temperature=1.0, top_k=0, top_p=0):
    model.eval()
    with torch.no_grad():
        # 处理输入
        input_indices = text_to_indices(prompt)[:-1]  # 去掉eos
        inputs = torch.LongTensor(input_indices).unsqueeze(1).to(device)
        
        # 逐步生成
        for _ in range(max_len):
            output = model(inputs)  # (seq_len, 1, vocab_size)
            
            # 取最后一个token的logits
            logits = output[-1, 0, :] / temperature
            
            # Top-k筛选
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')
                
            # Top-p筛选
            if top_p > 0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # 移除累计概率超过p的token
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                logits[sorted_indices[sorted_indices_to_remove]] = -float('Inf')
            
            # 采样
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 拼接结果
            inputs = torch.cat([inputs, next_token.unsqueeze(0)], dim=0)
            
            if next_token.item() == word2idx['<eos>']:
                break
                
        # 转换回文本
        output_indices = inputs.squeeze(1).cpu().tolist()
        return ' '.join([vocab[idx] for idx in output_indices])

# 生成示例
print("\n生成示例:")
print(generate("自然语言", temperature=0.8, top_k=2))
print(generate("深度学习", top_p=0.9))
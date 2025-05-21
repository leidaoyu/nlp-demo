import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import math
from models import MiniGPT
import json
from torch.utils.data import TensorDataset, DataLoader
# 修改为单独导入
from utils.encode import build_vocab, text_to_indices

# 加载词表
with open('vocab.txt', 'r', encoding='utf-8') as f:
    vocab = [w.strip() for w in f.readlines()]
word2idx = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)
print(f"词表大小: {vocab_size}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MiniGPT(vocab_size).to(device)


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
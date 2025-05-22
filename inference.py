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
import gradio as gr

# 加载词表
with open('vocab.txt', 'r', encoding='utf-8') as f:
    vocab = [w.strip() for w in f.readlines()]
word2idx = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)
print(f"词表大小: {vocab_size}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MiniGPT(vocab_size).to(device)
# 加载模型参数
model.load_state_dict(torch.load('ckpt/minigpt.pt', map_location=device))
model.eval()


# ================== 文本生成 ==================
def generate(prompt, max_len=20, temperature=1.0, top_k=0, top_p=0):
    model.eval()
    with torch.no_grad():
        # 处理输入
        input_indices = text_to_indices(prompt,word2idx)[:-1]  # 去掉eos
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
# print("\n生成示例:")
# print(generate("自然语言", temperature=0.8, top_k=2))
# print(generate("深度学习是", top_p=0.9))

def gradio_generate(prompt, max_len, temperature, top_k, top_p):
    return generate(prompt, max_len=max_len, temperature=temperature, top_k=top_k, top_p=top_p)

# ================== 文本向量与相似度 ==================
def get_text_embedding(text):
    """
    输入: 文本字符串
    输出: 该文本的embedding向量（对token embedding取均值）
    """
    model.eval()
    with torch.no_grad():
        input_indices = text_to_indices(text, word2idx)[:-1]  # 去掉eos
        input_tensor = torch.LongTensor(input_indices).to(device)
        # 取embedding层
        embeddings = model.embed(input_tensor)  # (seq_len, d_model)
        # 对所有token的embedding取均值，得到句子向量
        sent_vec = embeddings.mean(dim=0)  # (d_model,)
        return sent_vec.cpu().numpy()

def cosine_similarity(text1, text2):
    """
    输入: 两个文本
    输出: 余弦相似度
    """
    vec1 = get_text_embedding(text1)
    vec2 = get_text_embedding(text2)
    # 计算余弦相似度
    sim = (vec1 @ vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
    return float(sim)

import numpy as np

def gradio_similarity(text1, text2):
    sim = cosine_similarity(text1, text2)
    return f"余弦相似度: {sim:.4f}"

with gr.Blocks() as demo:
    with gr.Tab("文本生成"):
        gr.Markdown("# MiniGPT 文本生成 Demo")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="输入", value="")
                max_len = gr.Slider(5, 100, value=20, step=1, label="最大生成长度")
                temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.05, label="Temperature")
                top_k = gr.Slider(0, 20, value=0, step=1, label="Top-k (0为不启用)")
                top_p = gr.Slider(0, 1, value=0, step=0.01, label="Top-p (0为不启用)")
                btn = gr.Button("生成")
            with gr.Column():
                output = gr.Textbox(label="生成结果")
        btn.click(fn=gradio_generate, inputs=[prompt, max_len, temperature, top_k, top_p], outputs=output)

    # with gr.Tab("文本相似度"):
    #     gr.Markdown("# MiniGPT 文本向量相似度 Demo")
    #     text1 = gr.Textbox(label="文本1", value="")
    #     text2 = gr.Textbox(label="文本2", value="")
    #     sim_btn = gr.Button("计算相似度")
    #     sim_output = gr.Textbox(label="相似度结果")
    #     sim_btn.click(fn=gradio_similarity, inputs=[text1, text2], outputs=sim_output)

# 启动gradio界面
demo.launch()
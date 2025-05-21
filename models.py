import torch.nn as nn

# ================== 模型定义 ==================
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=2, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        
        # 简化版Transformer解码器层
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model*4
            ),
            num_layers=num_layers
        )
        
        self.fc = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # 输入形状: (seq_len, batch_size)
        x = self.embed(x)  # (seq_len, batch_size, d_model)
        
        # 生成注意力mask
        tgt_mask = nn.Transformer().generate_square_subsequent_mask(x.size(0)).to(x.device)
        
        # Transformer处理
        out = self.transformer(
            x, 
            memory=None,
            tgt_mask=tgt_mask
        )
        
        # 输出预测
        return self.fc(out)
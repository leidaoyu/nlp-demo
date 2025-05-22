import torch.nn as nn
import torch

# ================== 模型定义 ==================
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=2, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model*4
            ),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embed(x)  # (seq_len, batch_size, d_model)
        tgt_mask = nn.Transformer().generate_square_subsequent_mask(x.size(0)).to(x.device)
        # 构造 dummy memory
        memory = torch.zeros(x.size(0), x.size(1), x.size(2), device=x.device)
        out = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask
        )
        out = self.fc(out)
        return out  # (seq_len, batch_size, vocab_size)
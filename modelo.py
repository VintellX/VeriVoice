import torch
import torch.nn as nn
class VeriVoice(nn.Module):
    def __init__(self, mel_binos=80, embed_dim=128, num_heads=4, num_layers=2, num_classes=2, max_seq_len=1000):
        super().__init__()
        self.graph_conv = nn.Conv1d(mel_binos, mel_binos, kernel_size=3, padding=1, groups=mel_binos)
        self.frame_embed = nn.Linear(mel_binos, embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        encoder = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder, num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.max_seq_len = max_seq_len

    def forward(self, x):
        # x: (B, mel_binos, T)
        x = self.graph_conv(x)
        x = x.permute(0,2,1)
        T = x.size(1)
        x = self.frame_embed(x)
        if T > self.max_seq_len:
            x = x[:,:self.max_seq_len,:]; T = self.max_seq_len
        x = x + self.pos_emb[:,:T,:]
        x = x.permute(1,0,2)
        x = self.transformer(x)
        x = x.permute(1,0,2)
        x = x.mean(dim=1)
        return self.classifier(x)

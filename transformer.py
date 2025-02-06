import torch
import torch.nn as nn
import math


class InputBlock(nn.Module):

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)   

    def forward(self, x):
        return self.embedding(x)


class PositionalEncoding(nn.Module):

    def __init__(self, seq_len, d_model, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos_enc = torch.zeros(seq_len, d_model)
        
        pos = torch.arange(0, seq_len).unsqueeze(1)

        pos_enc[:,0::2] = torch.sin(pos/(torch.pow(10000, (torch.arange(0, d_model, 2)/d_model))))
        pos_enc[:,1::2] = torch.cos(pos/(torch.pow(10000, (torch.arange(1, d_model, 2)/d_model))))

        pos_enc = pos_enc.unsqueeze(0)

        self.register_buffer("pos_enc", pos_enc)

    def forward(self,x):
        
        x = x + self.pos_enc.requires_grad_(False)

        return self.dropout(x)

    
        








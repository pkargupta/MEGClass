import torch
from torch import nn
import sys


class MEGClassModel(nn.Module):

    def __init__(self, emb_dim, num_heads, dropout=0.1):
        #super().__init__(config)
        super(MEGClassModel, self).__init__()

        self.attention  = torch.nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
        # Two-layer MLP
        self.ffn1 = nn.Sequential(
            nn.Linear(emb_dim, 2*emb_dim),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(2*emb_dim, emb_dim)
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, emb_dim)
        )

        self.norm1 = nn.LayerNorm(emb_dim)

        self.sent_attention = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Tanh(),
            nn.Linear(emb_dim, emb_dim, bias=False)
            )
    
    def forward(self, input_emb, mask=None):
        # input_emb: batch size x sequence length x emb_dim
        X, _ = self.attention(input_emb, input_emb, input_emb, key_padding_mask=mask)
        X = X + input_emb
        X = self.ffn2(X)
        contextualized_sent = self.norm1(X) #[~mask] N x S x E

        # convert mask from N x S to N x S x E
        full_mask = (~mask).unsqueeze(-1).expand(X.size())
        exp_sent = torch.exp(self.sent_attention(contextualized_sent)) # N x S x E
        denom = torch.unsqueeze(torch.sum(exp_sent * (full_mask).int().float(), dim=1), dim=1) # N x 1 x E
        contextualized_doc = torch.sum((torch.div(exp_sent, denom) * contextualized_sent) * (full_mask), dim=1) # N x 1 x E

        return contextualized_sent, contextualized_doc
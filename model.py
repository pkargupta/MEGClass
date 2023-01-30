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

        self.scalar_sent_attention = nn.Sequential(
            nn.Linear(emb_dim, 1),
            nn.Tanh(),
            nn.Linear(1, 1, bias=False)
        )
    
    def forward(self, input_emb, mask=None):
        # input_emb: batch size x sequence length x emb_dim
        X, _ = self.attention(input_emb, input_emb, input_emb, key_padding_mask=mask)
        X = X + input_emb
        X = self.ffn2(X)
        contextualized_sent = self.norm1(X) #[~mask] N x S x E

        # scalar attention weight for each sentence
        exp_sent = torch.exp(self.scalar_sent_attention(contextualized_sent)) # N x S x 1
        exp_sent = torch.squeeze(exp_sent, dim=2) * (~mask).int().float() # N x S x 1 but all masked items are 0
        denom = torch.unsqueeze(torch.sum(exp_sent, dim=1), dim=1) # N x 1
        alpha = torch.unsqueeze(torch.div(exp_sent, denom), dim=2) # N x S x 1
        contextualized_doc = torch.sum(alpha.expand_as(contextualized_sent) * contextualized_sent, dim=1) # N x 1 x E

        # convert mask from N x S to N x S x E
        # full_mask = (~mask).unsqueeze(-1).expand(X.size())
        # exp_sent = torch.exp(self.sent_attention(contextualized_sent)) # N x S x E
        # denom = torch.unsqueeze(torch.sum(exp_sent * (full_mask).int().float(), dim=1), dim=1) # N x 1 x E
        # contextualized_doc = torch.sum((torch.div(exp_sent, denom) * contextualized_sent) * (full_mask), dim=1) # N x 1 x E

        return contextualized_sent, contextualized_doc, alpha
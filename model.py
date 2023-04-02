import torch
from torch import nn
import numpy as np
from tqdm import tqdm

def bertSentenceEmb(args, doc_to_sent, sent_repr):
    num_docs = len(doc_to_sent)
    doc_lengths = np.zeros(num_docs, dtype=int)
    padded_sent_repr = np.zeros((num_docs, args.max_sent, args.emb_dim))
    sentence_mask = np.ones((num_docs, args.max_sent))
    trimmed = 0


    for doc_id in tqdm(np.arange(num_docs)):
        start_sent = doc_to_sent[doc_id][0]
        end_sent = doc_to_sent[doc_id][-1]
        num_sent = end_sent - start_sent + 1
        if num_sent > args.max_sent:
            end_sent = start_sent + args.max_sent - 1
            num_sent = args.max_sent
            trimmed += 1
        embeddings = sent_repr[start_sent:end_sent+1]

        # save the number of sentences in each document
        doc_lengths[doc_id] = int(num_sent)
        # Add padded sentences
        padded_sent_repr[doc_id, :embeddings.shape[0], :] = embeddings
        # Update mask so that padded sentences are not included in attention computation
        sentence_mask[doc_id, :num_sent] = 0

    
    print(f"Trimmed Documents: {trimmed}")

    return padded_sent_repr, doc_lengths, sentence_mask

class MEGClassModel(nn.Module):
    def __init__(self, D_in, D_hidden, head, dropout=0.0):
        super(MEGClassModel, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=D_in, num_heads=head, dropout=dropout, batch_first=True)
        self.layernorm = nn.LayerNorm(D_in)
        self.embd = nn.Linear(D_in,D_hidden)
        self.attention = nn.Linear(D_hidden,1)
        
    def forward(self, x_org, mask=None):
        x, mha_w = self.mha(x_org,x_org,x_org,key_padding_mask=mask)
        x = self.layernorm(x_org+x)
        
        x = self.embd(x)
        x = torch.tanh(x) # contextualized sentences
        a = self.attention(x)
        if mask is not None:
            a = a.masked_fill_((mask == 1).unsqueeze(-1), float('-inf'))
        w = torch.softmax(a, dim=1) # alpha_k
        o = torch.matmul(w.permute(0,2,1), x) #doc 
        return o, mha_w, w, x # contextualized doc, multi-head attention weights, alpha_k, contextualized sent

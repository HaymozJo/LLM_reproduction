import torch.nn as nn
import torch.nn.functional as F
import torch


class Head(nn.Module):
    def __init__(self, mask, head_size, block_size, n_embd, device):
        super().__init__()
        self.Query = nn.Linear(n_embd, head_size) #We get the linear projections to the head_size
        self.Key = nn.Linear(n_embd, head_size)
        self.Value = nn.Linear(n_embd, head_size)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size, device = device))) # (T, T)

        self.mask = mask
        

    def forward(self, idx):
        _, T, d= idx.shape
        Q = self.Query(idx) #All (B, T, d)
        K = self.Key(idx)
        V = self.Value(idx)

        wei = Q@K.transpose(-2,-1) #(B, T, d) @ (B, d, T) ---> (B, T, T)

        wei = wei* (d **-0.5)
        #If we are in a decoder head
        if self.mask:
            wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        out = wei @ V #(B, T, T) @ (B, T, d) --> (B, T, d)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd *4), #as in the paper, multiplied by 4
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd)
        )
        self.ln = nn.LayerNorm(n_embd)


    def forward(self, idx):
        out = self.net(idx)
        out = self.ln(idx + out)
        return out
    
class EncodeBlock(nn.Module):
    def __init__(self, block_size, n_embd, num_heads, device = 'cpu' ):
        super().__init__()
        head_size = n_embd//num_heads
        #Our list of heads 
        self.multiHeads = nn.ModuleList(Head(False, head_size, block_size, n_embd, device=device) for _ in range(num_heads))
        self.proj = nn.Linear(n_embd, n_embd)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        #Note the x here for residual pathway
        out = torch.cat([head(x) for head in self.multiHeads], dim = -1) #(B, T, d*h) == (B, T, C) 
        out = self.proj(out) 
        out1 = self.ln1(x + out) #kind of a checkpoint for residual path
        out = self.ffwd(out1)
        out = self.ln2(out1 + out) 
        return out

class AIAYNModel(nn.Module):
    def __init__(self, n_embd, voc_size ,block_size, n_heads, n_layers, device = 'cpu'):
        super().__init__()
        self.encode_embedding = nn.Embedding(voc_size, n_embd) #(B, T, vocab_size) --> (B, T, C)
        self.decode_embedding = nn.Embedding(voc_size, n_embd) #(B, T, vocab_size) --> (B, T, C) 
        self.positional_embedding = nn.Embedding(block_size, n_embd) #(B, T, vocab_size) --> (B, T, C)
        self.Encodeblocks = nn.Sequential(*[EncodeBlock(block_size, n_embd, n_heads, device) for _ in range(n_layers)])
        #Save for other fct:
        self.n_embd = n_embd #C
        self.voc_size = voc_size #vocab_size
        self.block_size = block_size #T
        self.device = device

    def forward(self, idx):
        token_embd = self.encode_embedding(idx) #(B, T, C)
        pos_embd = self.positional_embedding(torch.arange(self.block_size, device=self.device)) # (T, C)
        
        embd = token_embd + pos_embd #(B, T, C)
        out = self.Encodeblocks(embd)
        return out
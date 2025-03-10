import torch.nn as nn
import torch
from torch.nn import functional as F

class Head(nn.Module):
    def __init__(self, head_size, block_size, n_embd, device, dropout = 0.1):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril',  torch.tril(torch.ones(block_size, block_size, device = device)))
        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size #d
        self.block_size = block_size #T
        self.n_embd = n_embd #C
        

    def forward(self, x):
        B, T, C = x.shape #Here it arrives already embedded in (B,T, C)
        k = self.key(x)   # (B, T, d)
        q = self.query(x) # (B, T, d)
        v = self.value(x) # (B, T, d)

        wei =  q @ k.transpose(-2, -1) # (B, T, d) @ (B, d, T) ---> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) 
        wei = self.dropout(wei)
        
        out = wei @ v #(B, T, T) @ (B, T, d) --> (B, T, d)
        return out
    

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, block_size, n_embd, device,  dropout=0.1):
        super().__init__()
        self.heads = nn.ModuleList(Head(head_size, block_size, n_embd, device=device)
                                    for _ in range(num_heads)) # h heads of (B, T, d)
        self.proj = nn.Linear(n_embd, n_embd)
        self.device = device
        self.dropout = nn.Dropout(dropout)

    def forward(self, idx):    
        out = torch.cat([head(idx) for head in self.heads], dim = -1) #h *(B, T, d) 
        out = self.proj(out) # #projection back into residual pathway
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), #projection layer back into residual pathway
            nn.LayerNorm(n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_heads, block_size, n_embd, device, dropout = 0.1):
        super().__init__()
        self.head_size = n_embd//n_heads
        self.MultiHeads = MultiHeadAttention(n_heads, self.head_size, block_size, n_embd, device, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.MultiHeads(self.ln1(x)) #Note: we added addition of the x to have a residual pathway 
        x = x +self.ffwd(self.ln2(x)) # We also added the LayerNorms in a prenorm formulation
        return x

class MultiHeadAttentionModel(nn.Module):
    def __init__(self, n_heads, block_size, n_embd, vocab_size, n_layers, device, dropout=0.1):
        super().__init__()
        #embeddings:
        self.token_embedding = nn.Embedding(vocab_size, n_embd) #(C, C)
        self.positional_embedding = nn.Embedding(block_size, n_embd) #(T, C)
        self.blocks = nn.Sequential(*[Block(n_heads, block_size, n_embd, device, dropout) for _ in range(n_layers)])
        self.lm_head =nn.Linear(n_embd, vocab_size) # C --> voc_size


        self.device = device
        self.block_size = block_size #T
        self.n_embd = n_embd #C
        self.vocab_size = vocab_size #voc_size
    
    def forward(self, idx, targets = None):
        B, T = idx.shape #in the model it arrives in a (B, T)
        token_embd = self.token_embedding(idx) #(B, T, C)
        pos_embd = self.positional_embedding(torch.arange(T, device=self.device)) #(T, C)
        x = token_embd + pos_embd #(B, T, C)
        x = self.blocks(x)
        logits = self.lm_head(x) #(B, T, voc_size)
        if targets is None:
            loss = None
        else:
            logits = logits.reshape(B*T, -1) # (B*T, voc_size)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            context_idx = idx[:, -self.block_size:]

            logits, loss = self(context_idx)
            # focus only on last token:
            logits = logits[:, -1, :]
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, voc_size)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
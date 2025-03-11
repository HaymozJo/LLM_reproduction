import torch.nn as nn
import torch
from torch.nn import functional as F

#Head for selfAttention with no multihead in sight -> headsize == n_embd
class Head(nn.Module):
    def __init__(self, n_embd, block_size, device):
        super().__init__()
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.register_buffer('tril',  torch.tril(torch.ones(block_size, block_size, device = device)))
        self.block_size = block_size #T
        self.n_embd = n_embd #C
        

    def forward(self, x):
        B, T, C = x.shape #Here it arrives already embedded in (B,T, C)
        k = self.key(x)   # (B, T, d)
        q = self.query(x) # (B, T, d)
        v = self.value(x)

        wei =  q @ k.transpose(-2, -1) # (B, T, d) @ (B, d, T) ---> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        
        out = wei @ v
        return out
    

class SelfAttentionModel(nn.Module):
    def __init__(self, n_embd, block_size, vocab_size, device):
        super().__init__()
        #embeddings:
        self.token_embedding = nn.Embedding(vocab_size, n_embd) #(C, C)
        self.positional_embedding = nn.Embedding(block_size, n_embd) #(T, C)
        self.head = Head(n_embd, block_size, device= device) # (B, T, C) -> (B, T, C)
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
        x = self.head(x) #Apply the head --> (B, T, d)

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

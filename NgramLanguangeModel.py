import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)


class NgramLanguageModel(nn.Module):

    def __init__(self, vocab_size, Ngram = 2): # Default to bigram (N=2)
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        self.N = Ngram-1
        
        #projection of the linear model into the vocab size from N*C to C
        if self.N != 1: #Bigram case
            self.lm_context_to_logit = nn.Linear(vocab_size*self.N, vocab_size)
        else: self.lm_context_to_logit = nn.Identity()

    def forward(self, idx, targets=None):
        
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)
        B, T, C = logits.shape
        if T < self.N:
            raise ValueError(f"Sequence length must be at least {self.N}for trigram model")
        
        logits = torch.zeros((B, T-self.N, C), device=idx.device)

        for t in range(T-self.N):
            context_idx = idx[:, t: t+self.N]
            context_emb = self.token_embedding_table(context_idx)  # (B, N, C)  
            # Flatten the context embeddings
            context_flat = context_emb.reshape(B, -1)  # (B, N*C)
            
            # Get logits through the projection
            logits[:, t] = self.lm_context_to_logit(context_flat) 
        
        if targets is None:
            loss = None
        else:
            shifted_targets = targets[:, self.N:]  #(B, T-N)
            targets = shifted_targets.reshape(-1) # (B*(T-N))
            logits = logits.reshape(B*(T-self.N), -1) # (B*(T-N), C)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
        

        

    def generate(self, idx, max_new_tokens):
        B, T = idx.shape
        #Keepsafe if block size < context size
        if T < self.N: 
                print("sequence not big enough to start generation, added a random padding at the start")
                # Add dummy tokens to build context if needed, use what would come out of our feature as high
                padding_needed = self.N - T
                random_padding = torch.randint(0, self.lm_context_to_logit.out_features, 
                                            (B, padding_needed), device=idx.device)
                idx = torch.cat((random_padding, idx), dim=1)
        for _ in range(max_new_tokens):
            # Here we basically don't call the whole model as we do not need logits for everything
            # (in Bigram same, but logic applied was more for a transformer type of gen)
            # So we only process through our context and get the logits for our context
            # => way faster generation
            context_idx = idx[:, -self.N:]
            context_emb = self.token_embedding_table(context_idx)  # (B, N, C)  
            # Flatten the context embeddings
            context_flat = context_emb.reshape(B, -1)  # (B, N*C)
            
            # Get logits through the projection
            logits = self.lm_context_to_logit(context_flat) 
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx



import torch.nn as nn
import torch.nn.functional as F
import torch


class Head(nn.Module):
    def __init__(self, mask, head_size, block_size, n_embd, device, cross = False):
        super().__init__()
        self.Query = nn.Linear(n_embd, head_size) #We get the linear projections to the head_size

        self.Key = nn.Linear(n_embd, head_size)
        self.Value = nn.Linear(n_embd, head_size)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size, device = device))) # (T, T)

        self.mask = mask
        self.cross = cross

        self.head_size = head_size
    

    """
    param:
        * idx : (B, T, C)
        * K_encod: (B, T, d)
        * V_encod: (B, T, d)
    
    """
    def forward(self, idx, K_encod = None, V_encod = None):
        _, T, _ = idx.shape
        Q = self.Query(idx) #All (B, T, d)
        if self.cross:
            K = self.Key(K_encod) #Tensors from the final layer of encoder, projected into the correct value
            V = self.Value(V_encod)
        else:
            K = self.Key(idx)
            V = self.Value(idx)

        wei = Q@K.transpose(-2,-1) #(B, T, d) @ (B, d, T) ---> (B, T, T)

        wei = wei* (self.head_size **-0.5)
        #If we are in a decoder head
        if self.mask:
            wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        out = wei @ V #(B, T, T) @ (B, T, d) --> (B, T, d)
        return out, K, V


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd *4), #as in the paper, multiplied by 4
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )
        self.ln = nn.LayerNorm(n_embd)

    def forward(self, idx):
        out = self.net(idx)
        out = self.ln(idx + out)
        return out
    
class EncodeBlock(nn.Module):
    def __init__(self, block_size, n_embd, num_heads, device = 'cpu', dropout=0.2):
        super().__init__()
        head_size = n_embd//num_heads
        #Our list of heads 
        self.multiHeads = nn.ModuleList(Head(False, head_size, block_size, n_embd, device=device) for _ in range(num_heads))
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        #Note the x here for residual pathway
        head_outputs = [head(x) for head in self.multiHeads] #(B, T, d*h, (TUPLE)) == (B, T, C, (TUPLE)) 
        #Get the different outputes, here K,V are only really used when our final block is reached
        out = torch.cat([h[0] for h in head_outputs], dim=-1) #(B, T, C)
        K = torch.cat([h[1] for h in head_outputs], dim=-1)
        V = torch.cat([h[2] for h in head_outputs], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)  # Add dropout after projection
        out1 = self.ln1(x + out) #kind of a checkpoint for residual path
        out = self.ffwd(out1)
        out = self.ln2(out1 + out) 
        return out, K, V


class DecodeBlock(nn.Module):
    def __init__(self, block_size, n_embd, num_heads, device = 'cpu', dropout=0.2):
        super().__init__()
        head_size = n_embd//num_heads
        #Our list of heads 
        #first one with masking
        self.multiHeadsMasked = nn.ModuleList(Head(True, head_size, block_size, n_embd, device=device) for _ in range(num_heads))
        #Second one with the input of K,V
        self.multiHeadsCross =  nn.ModuleList(Head(False, head_size, block_size, n_embd, cross=True, device=device) for _ in range(num_heads)) 
        #The projections, one and two after multiheads, las proj (last_proj) at the end
        self.proj1 = nn.Linear(n_embd, n_embd)
        self.proj2 = nn.Linear(n_embd, n_embd)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        #Feed-Forward neural network
        self.ffwd = FeedForward(n_embd, dropout)
        #The 3 layerNorms
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)

    def forward(self, idx, K_enc, V_enc):
        heads_output  = [h(idx) for h in self.multiHeadsMasked]
        out = torch.cat([h[0] for h in heads_output], dim = -1) #In the decoder we don't care about it's K and V
        out = self.proj1(out)
        out = self.dropout1(out)
        out1 = self.ln1(idx + out) #We keep the out1 info for futre residual path
        
        crossHeads_output = [h(out1, K_enc, V_enc) for h in self.multiHeadsCross]
        out_cross = torch.cat([h[0] for h in crossHeads_output], dim = -1)
        out = self.proj2(out_cross)
        out = self.dropout2(out)
        out2 = self.ln2(out1 + out)

        out = self.ffwd(out2)
        out = self.ln3(out2 + out)
        return out


class AIAYNModel(nn.Module):
    def __init__(self, n_embd, voc_size ,block_size, n_heads, n_layers, device = 'cpu'):
        super().__init__()
        self.encode_embedding = nn.Embedding(voc_size, n_embd) #(B, T, vocab_size) --> (B, T, C)
        self.decode_embedding = nn.Embedding(voc_size, n_embd) #(B, T, vocab_size) --> (B, T, C) 
        self.positional_embedding = nn.Embedding(block_size, n_embd) #(B, T, vocab_size) --> (B, T, C)
        self.EncodeBlocks = nn.Sequential(*[EncodeBlock(block_size, n_embd, n_heads, device) for _ in range(n_layers)])
        self.DecodeBlocks = nn.Sequential(*[DecodeBlock(block_size, n_embd, n_heads, device) for _ in range(n_layers)])
        #last proj:
        self.lm_head = nn.Linear(n_embd, voc_size)
        #Save for other fct:
        self.n_embd = n_embd #C
        self.voc_size = voc_size #vocab_size
        self.block_size = block_size #T
        self.device = device
        # Special tokens
        self.BOS_TOKEN_ID = 100  # BERT's [CLS] token
        self.EOS_TOKEN_ID = 102  # BERT's [SEP] token
        self.PAD_TOKEN_ID = 0    # BERT's [PAD] token

    """
    Param:
        * src: (B, T, C) 
    """ 
    def _encoder_forward(self, src):
        
        token_embd = self.encode_embedding(src) #(B, T, C)
        # Create position indices and clamp them to block_size - 1
        positions = torch.arange(src.size(1), device=self.device)
        positions = torch.clamp(positions, max=self.block_size - 1)
        pos_embd = self.positional_embedding(positions) # (T, C)
        
        x_enc = token_embd + pos_embd #(B, T, C)
                # Process through encoder blocks
        encoder_K = None
        encoder_V = None
        for block in self.EncodeBlocks:
            x_enc, K, V = block(x_enc)
            encoder_K = K  # We'll use the K from the last encoder block
            encoder_V = V  # We'll use the V from the last encoder block

        return x_enc, encoder_K, encoder_V
    
    """
    param:
        * x_dec : (B, T', voc_size)
        * K_encod, V_encod: (B, T, C)
    """
    def _decoder_forward(self, x_dec, K_encod, V_encod):
        
        token_embd = self.decode_embedding(x_dec)

        # Create position indices and clamp them to block_size - 1
        positions = torch.arange(x_dec.size(1), device=self.device)
        positions = torch.clamp(positions, max=self.block_size - 1)

        pos_embd = self.positional_embedding(positions)
        x_dec = token_embd + pos_embd

        for block in self.DecodeBlocks:
            x_dec = block(x_dec, K_encod, V_encod)

        return self.lm_head(x_dec)
    
    """
    params:
        *src : (B, T', C)?
    """
    def forward(self, src, tgt = None):
        B, _ = src.shape
        _, K_encod, V_encod = self._encoder_forward(src)
        
        if tgt is not None:
            B, T_tgt = tgt.shape
            # Add BOS token to decoder input
            decoder_input = torch.cat([
                torch.full((B, 1), self.BOS_TOKEN_ID, dtype=tgt.dtype, device=tgt.device),
                tgt[:, :-1]  # Remove last token for teacher forcing
            ], dim=1)
            
            logits = self._decoder_forward(decoder_input, K_encod, V_encod)
            logits = logits[:, :-1, :].reshape(-1, self.voc_size)
            targets = tgt[:, 1:].reshape(-1).long()
            
            if logits.size(0) != targets.size(0):
                raise ValueError(f"Shape mismatch: logits {logits.shape}, targets {targets.shape}")
            
            loss = F.cross_entropy(logits, targets)
        else:
            logits = self.generate(src, K_encod, V_encod)
            loss = None

        return logits, loss
    
    def generate(self, src, K_encod = None, V_encod = None, max_token = 100, temperature = 0.7, top_k = 50):
        if K_encod is None or V_encod is None:
            _, K_encod, V_encod = self._encoder_forward(src)

        B = src.size(0)
        generated = torch.full((B, 1), self.BOS_TOKEN_ID, dtype=torch.long, device=self.device)
        
        # Track if each sequence has generated an EOS token
        finished = torch.zeros(B, dtype=torch.bool, device=self.device)
        
        for _ in range(max_token - 1):
            # If all sequences are finished, break
            if finished.all():
                break
                
            logits = self._decoder_forward(generated, K_encod, V_encod)
            logits = logits[:, -1, :] / temperature  # Apply temperature
            
            # Apply top-k sampling
            top_k_logits, top_k_indices = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
            probs = F.softmax(top_k_logits, dim=-1)
            
            # Sample from the filtered distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            idx_next = torch.gather(top_k_indices, 1, idx_next)
            
            # Append sampled index to the running sequence
            generated = torch.cat((generated, idx_next), dim=1)
            
            # Check for EOS tokens
            finished = finished | (idx_next.squeeze() == self.EOS_TOKEN_ID)
        
        return generated

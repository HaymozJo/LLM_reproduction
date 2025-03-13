import torch.nn as nn
import torch.nn.functional as F
import torch


class Head(nn.Module):
    def __init__(self, mask, head_size, block_size, n_embd, device, cross = False):
        super().__init__()
        self.Query = nn.Linear(n_embd, head_size) #We get the linear projections to the head_size
        if cross:
            pass
        else:
            self.Key = nn.Linear(n_embd, head_size)
            self.Value = nn.Linear(n_embd, head_size)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size, device = device))) # (T, T)

        self.mask = mask
        self.cross = cross

        self.head_size = head_size
        

    def forward(self, idx, K_encod = None, V_encod = None):
        _, T, _ = idx.shape
        Q = self.Query(idx) #All (B, T, d)
        if self.cross:
            K = K_encod #Tensors from the final layer of encoder
            V = V_encod
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
        head_outputs = [head(x) for head in self.multiHeads] #(B, T, d*h, (TUPLE)) == (B, T, C, (TUPLE)) 
        #Get the different outputes, here K,V are only really used when our final block is reached
        out = torch.cat([h[0] for h in head_outputs], dim=-1) #(B, T, C)
        K = torch.cat([h[1] for h in head_outputs], dim=-1)
        V = torch.cat([h[2] for h in head_outputs], dim=-1)
        out = self.proj(out) 
        out1 = self.ln1(x + out) #kind of a checkpoint for residual path
        out = self.ffwd(out1)
        out = self.ln2(out1 + out) 
        return out, K, V 


class DecodeBlock(nn.Module):
    def __init__(self, block_size, n_embd, num_heads, device = 'cpu' ):
        super().__init__()
        head_size = n_embd//num_heads
        #Our list of heads 
        #first one with masking
        self.multiHeadsMasked = nn.ModuleList(Head(True, head_size, block_size, n_embd, device=device) for _ in range(num_heads))
        #Second one with the input of K,V
        self.multiHeadsCross =  nn.ModuleList(Head(True, head_size, block_size, n_embd, cross=True, device=device) for _ in range(num_heads)) 
        #The projections, one and two after multiheads, las proj (last_proj) at the end
        self.proj1 = nn.Linear(n_embd, n_embd)
        self.proj2 = nn.Linear(n_embd, n_embd)
        #Feed-Forward neural network
        self.ffwd = FeedForward(n_embd)
        #The 3 layerNorms
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)

    def forward(self, idx, K_enc, V_enc):
        heads_output  = [h(idx) for h in self.multiHeadsMasked]
        out = torch.concat([h[0] for h in heads_output], dim = -1) #In the decoder we don't care about it's K and V
        out = self.proj1(out)
        out1 = self.ln1(idx + out) #We keep the out1 info for futre residual path
        
        crossHeads_output = [h(out1, K_enc, V_enc) for h in self.multiHeadsCross]
        out_cross = torch.cat([h[0] for h in crossHeads_output], dim = -1)
        out = self.proj2(out_cross)
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

    def _encoder_forward(self, src):
        token_embd = self.encode_embedding(src) #(B, T, C)
        pos_embd = self.positional_embedding(torch.arange(src.size(1), device=self.device)) # (T, C)
        
        x_enc = token_embd + pos_embd #(B, T, C)
                # Process through encoder blocks
        encoder_K = None
        encoder_V = None
        for block in self.Encodeblocks:
            x_enc, K, V = block(x_enc)
            encoder_K = K  # We'll use the K from the last encoder block
            encoder_V = V  # We'll use the V from the last encoder block

        return x_enc, encoder_K, encoder_V

    def _decoder_forward(self, x_dec, K_encod, V_encod):
        
        token_embd = self.decode_embedding(x_dec)
        pos_embd = self.positional_embedding(torch.arange(x_dec.size(1), device=self.device))
        x_dec = token_embd + pos_embd

        for block in self.DecodeBlocks:
            out = block(x_dec, K_encod, V_encod)

        return self.lm_head(out)
    
    def forward(self, src, tgt = None):
        B, T, _ = src.shape
        _, K_encod, V_encod = self._encoder_forward(src)
        #Case we have the targets, we want to get the logits and the loss to evaluate the model
        if tgt is not None:
            out_dec = self._decoder_forward(tgt, K_encod, V_encod)
            #calculate loss
            loss = None
            logits = logits.reshape(B*T, -1) # (B*T, voc_size)
            tgt = tgt.view(B*T)
            loss = F.cross_entropy(logits, tgt)
        
        #Case we do not have the targets, we want to generate the code
        else:
            out_dec = self.generate(src, K_encod, V_encod)
            loss = None

        return out_dec, loss
    
    def generate(self, src, K_encod = None, V_encod = None, max_token = 100):
        #If no K or V, generate new ones (Normally won't happen)
        if K_encod is None or V_encod is None:
            _, K_encod, V_encod = self._encoder_forward(src)

        B = src.size(0)
        # Bert BOS = 100, XLNET = 1
        BOS_TOKEN_ID = 100  # Replace with your actual BOS token ID
        
        #Start with BOS
        generated = torch.ones((B, 1), dtype=torch.long, device=self.device) * BOS_TOKEN_ID

        for _ in range(max_token -1):
            logits = self._decoder_forward(generated, K_encod, V_encod)
            logits = logits[:, -1, :]
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, voc_size)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            generated = torch.cat((generated, idx_next), dim=1) # (B, T+1)
        
        return generated

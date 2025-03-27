import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import os
from torch.nn.utils.rnn import pad_sequence 
from AIAYNModel import AIAYNModel
from torchinfo import summary
import numpy as np
from Helpers import Encoding
#Hyperparameters------------------------------
path_data = "Data/Bert.parquet"
split = 0.2 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
encoding_name = 'Bert' #Choose encoding in ['Bert', 'XLNet', 'Int'] following your preprocessing choice
vocab_size =  30522   # Size of BERT's vocabulary  --> vocab_size
block_size = 32 #Set at 32 given the exploration notebook
n_embd = 252 # number of embeddings, must be divisible by n_head (6 * 42, smaller than the paper) --> C
batch_size = 32 #number of batches dealt at the same time --> B
n_head = 6 #numbers of heads processed in parallel in a multihead block --> n_head or h
n_layer = 6 # number of repeated block one after the other (6 in paper ) --> n_layer

#Training parameters
learning_rate = 1e-4 
weight_decay = 0.01  # weight decay
warmup_steps = 1000  # Number of warmup steps
max_iters = 50000  # Total number of training iterations
eval_iters = 500 #each eval_iter we evaluate the loss

#Saving parameters
path_save_model = "Models/"
#---------------------------------------------

df = pd.read_parquet(path_data)

#Assert if we have all strings bigger than block size to ensure our context takes everything
if df["input_token"].str.len().min() < block_size:
    raise OverflowError("Block size is too big and all the samples will not fit")

#separate by li

train_df, test_df = train_test_split(df, test_size=split)
#get batch function 
#Note: may be good to put in Helpers (dataLoader?)
train_input_sequences = [torch.tensor(row['input_token'], dtype = torch.int) for _, row in train_df.iterrows()]
train_target_sequences = [torch.tensor(row['target_token'], dtype = torch.int) for _, row in train_df.iterrows()]
test_input_sequences = [torch.tensor(row['input_token'], dtype = torch.int) for _, row in test_df.iterrows()]
test_target_sequences = [torch.tensor(row['target_token'], dtype = torch.int) for _, row in test_df.iterrows()]

print(df["input_token"].str.len().max())
print(df["input_token"].str.len().min())
print(df["target_token"].str.len().max())
print(df["target_token"].str.len().min())


# Initialize model and optimizer
m = AIAYNModel(n_embd, vocab_size, block_size, n_head, n_layer, device=device)
m = m.to(device)  # Move model to device

# Initialize optimizer with weight decay
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate, weight_decay=weight_decay)

#We get lists of tensors for each input/target tokens. the tensors size change depending on the input/target as
#all songs do not have zthe same length
def get_batch(train = True, device = 'cpu'):
    inputs_sequences = train_input_sequences if train else test_input_sequences
    target_sequences = train_target_sequences if train else test_target_sequences

    text_indices = torch.randint(len(inputs_sequences), (batch_size,)) #B rows selected randomly
    encoder_inputs, decoder_trgts = [], []
    
    # Get actual lengths for this batch
    for idx in text_indices:
        in_seq = inputs_sequences[idx]
        trgt_seq = target_sequences[idx]
        
        # Take the actual sequence length, up to block_size
        in_len = min(len(in_seq), block_size)
        trgt_len = min(len(trgt_seq), block_size)
        
        # Take only the needed portion
        encoder_inputs.append(in_seq[:in_len])
        decoder_trgts.append(trgt_seq[:trgt_len])
    
    # Pad sequences in batch to longest sequence
    encoder_in = pad_sequence(encoder_inputs, batch_first=True, padding_value=m.PAD_TOKEN_ID)
    decoder_trgt = pad_sequence(decoder_trgts, batch_first=True, padding_value=m.PAD_TOKEN_ID)
    
    return encoder_in.to(device), decoder_trgt.to(device)

@torch.no_grad()  # Disable gradients
def estimate_loss():
    out = {}
    m.eval() #explain which mode we are in. good practice, not useful here
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(train=(split == 'train'), device=device)
            _, loss = m(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()#back to training mode after. 
    return out

encoder_in, decoder_trgt = get_batch(device=device)
print(f"shapes are: enc_in {encoder_in.shape}, dec_trgt {decoder_trgt.shape}")


# Learning rate scheduler with warmup
def get_lr(step):
    if step < warmup_steps:
        return learning_rate * (step / warmup_steps)
    return learning_rate

train_losses, test_losses = [], []
for steps in range(max_iters):
    # Update learning rate
    lr = get_lr(steps)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    if steps % eval_iters == 0: 
        losses = estimate_loss()
        train_losses.append(losses['train'].item())
        test_losses.append(losses['test'].item())
        print(f"Step [{steps}/{max_iters}], lr: {lr:.2e}, losses: [Train: {losses['train']:.5f}, Test: {losses['test']:.5f}]")
        
        # Get the current batch
        xb, yb = get_batch(train=True, device=device)
        
        # Get model predictions
        logits, loss = m(xb, yb)
        
        # Calculate prediction distribution
        pred_tokens = torch.argmax(logits.reshape(-1, vocab_size), dim=-1)
        
        # Print diagnostics
        print(f"\nStep {steps} diagnostics:")
        print(f"Loss: {loss.item():.4f}")
        
        # Count token types
        n_pad = (pred_tokens == m.PAD_TOKEN_ID).sum().item()
        n_total = pred_tokens.numel()
        print(f"Padding tokens: {n_pad}/{n_total} ({n_pad/n_total*100:.1f}%)")
        
        # Print a sample translation
        sample_idx = torch.randint(0, batch_size, (1,)).item()
        encoding = Encoding()
        #TODO: change to the encoding we are using
        encode, decode = encoding.BertToken(size_df=1, progress=False)
        input_text = decode(xb[sample_idx].tolist())
        target_text = decode(yb[sample_idx].tolist())
        pred_text = decode(torch.argmax(logits[sample_idx], dim=-1).tolist())
        print(f"\nSample translation:")
        print(f"Input:  {input_text}")
        print(f"Target: {target_text}")
        print(f"Pred:   {pred_text}")
    
    # sample a batch of data
    xb, yb = get_batch(train=True, device=device)

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
    optimizer.step()

#Save model
name = f"{encoding_name}_MI{max_iters}_lr{learning_rate}_wd{weight_decay}"
path_dir = path_save_model + name
os.mkdir(path_dir)
#Save everything in the models file
path = path_dir + "/" + name + ".pt"
torch.save(m, path)
#store losses
array = np.array([train_losses, test_losses])
np.savetxt(path_dir + "/" + name + "losses.csv", array, delimiter=",")
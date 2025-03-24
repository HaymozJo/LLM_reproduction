import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.nn.utils.rnn import pad_sequence 
from AIAYNModel import AIAYNModel
from torchinfo import summary
#Hyperparameters------------------------------
path_data = "Data/Bert.parquet"
split = 0.2 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
encoding = 'Bert' #Choose encoding in ['Bert', 'XLNet', 'Int'] following your preprocessing choice
vocab_size =  30522   # Size of BERT's vocabulary  --> vocab_size
block_size = 32 #Set at 32 given the exploration notebook
n_embd = 60 # Change it to the attention paper when finished --> C
batch_size = 32 #number of batches dealt at the same time --> B
n_head = 6 #numbers of heads processed in parallel in a multihead block --> n_head or h
n_layer = 3 # number of repeated block one after the other (6 in paper ) --> n_layer

#Training parameters
learning_rate = 3e-4
max_iters = 5000  # Total number of training iterations
eval_iters = 500
max_eval_iters = 5000
max_tokens = 9000

#Saving parameters
path_save_model = "Models/"
#---------------------------------------------

df = pd.read_parquet(path_data)

#Assert if we have all strings bigger than block size to ensure our context takes everything
#Note: Padding? Trimming?
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
#We get lists of tensors for each input/target tokens. the tensors size change depending on the input/target as
#all songs do not have zthe same length
def get_batch(train = True, device = 'cpu'):
    # Randomly select indices for the rows
    inputs_sequences = train_input_sequences if train else test_input_sequences
    target_sequences = train_target_sequences if train else test_target_sequences

    text_indices = torch.randint(len(inputs_sequences), (batch_size,)) #B rows selected randomly
    encoder_inputs, decoder_trgts = [], []
    for idx in text_indices:
        in_seq = inputs_sequences[idx]
        trgt_seq = target_sequences[idx]
        
        #Check our input sequences make sense, otherwise we may have to kick them 
        if len(in_seq) < block_size or len(trgt_seq) < block_size:
            raise ValueError(f"Sequence too short. Input: {len(in_seq)}, Target: {len(trgt_seq)}")
            
        encoder_inputs.append(in_seq)
        decoder_trgts.append(trgt_seq)
    
    encoder_in = torch.stack(encoder_inputs) #(B,T)
    decoder_trgt = torch.stack(decoder_trgts) #(B,T)
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

# Initialize model and optimizer
m = AIAYNModel(n_embd, vocab_size, block_size, n_head, n_layer, device=device)
m = m.to(device)  # Move model to device
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

train_losses, test_losses = [], []
for steps in range(max_iters):
    
    if steps % eval_iters == 0: 
        losses = estimate_loss()
        train_losses.append(losses['train'].item())
        test_losses.append(losses['test'].item())
        print(f"Step [{steps}/{max_iters}], losses: [Train: {losses['train']:.5f}, Test: {losses['test']:.5f}]")
    
    # sample a batch of data
    xb, yb = get_batch(train=True, device=device)

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#Save model
path_dir = path_save_model + f"AIAYN_MI{max_iters}_lr{learning_rate}"
os.mkdir(path_dir)
#Save everything in the models file
path = path_dir + "/" + name + ".pt"
torch.save(m, path)
array = np.array([train, test])
np.savetxt(path_dir + "/" + name + "losses.csv", array, delimiter=",")
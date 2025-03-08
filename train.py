import torch
import torch.nn as nn
from torch.nn import functional as F
from BigramLanguangeModel import BigramLanguageModel


torch.manual_seed(1337)

#hyperparameters
batch_size = 32 # number indep sequence processed in parallel
block_size = 8 #maximum context lenght
max_iters = 10000
eval_interval = 200
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32

#Get text and size
with open("input.txt", 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)

#Encoding and decoding functions for char -> integer
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

#Encode the text into a tensor
encoded = encode(text)
data = torch.tensor(encoded, dtype=torch.long)

#separate train and test data
n = int(0.9*len(data))
train_data = data[:n]
test_data = data[n:]

#Helper to get batch
def get_batch(split):
  data = train_data if split == 'train' else test_data
  ix = torch.randint(len(data) - block_size, (batch_size,))#generate random offsets, start of our different block sequences
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])

  x,y = x.to(device), y.to(device)
  return x, y


#Instantiate model and attach an adam optimizer to it
m = BigramLanguageModel(vocab_size)
m = m.to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr = learning_rate)

@torch.no_grad()  # Disable gradients
def estimate_loss():
    out = {}
    m.eval() #explain which mode we are in. good practice, not useful here
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = m(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()#back to training mode after. 
    return out

for steps in range(max_iters):

    if steps%eval_iters == 0: 
        losses = estimate_loss()
        print("Epoch [{}/{}], losses: [Train: {}, Test: {}".format(steps, max_iters, losses['train'], losses['test']))
    
    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    



idx = torch.zeros((1, 1), dtype = torch.long, device=device) #== '/n'

max_new_tokens = 300
pred = m.generate(idx, max_new_tokens)

print(decode(pred[0].tolist()))








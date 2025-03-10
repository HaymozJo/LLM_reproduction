import torch
import torch.nn as nn
from torch.nn import functional as F
from NgramLanguangeModel import NgramLanguageModel
from BigramLanguageModel import BigramLanguageModel
from SelfAttention import SelfAttentionModel
import os
import numpy as np

#hyperparameters
path_data = "Data/"
path_input = path_data + "input.txt"
path_save_model = "Models/"
batch_size = 32 # number indep sequence processed in parallel
block_size = 8 #maximum context lenght
head_size = 8 #if solo keep it = block size, if multi better to blockSize/#heads
n_embd = 32
max_iters = 20000
eval_interval = 200
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
Ngram = 7
model_choice = 'SelfAttention' #choice of model, bigram  base = "Bigram", ngram = 'Ngram', 1 head = 'SelfAttention

#Get text and size
with open(path_input, 'r', encoding='utf-8') as f:
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

if model_choice == 'Bigram':
    m = BigramLanguageModel(vocab_size)
elif model_choice == 'Ngram':
    m = NgramLanguageModel(vocab_size, Ngram = Ngram)
elif model_choice == 'SelfAttention':
    m = SelfAttentionModel(head_size, block_size, n_embd, vocab_size, device=device)
else:
    raise Exception("The model name set is false")

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

train, test = [], []
for steps in range(max_iters):
    
    if steps%eval_iters == 0: 
        losses = estimate_loss()
        train.append(losses['train'].item())
        test.append(losses['test'].item())
        print("Epoch [{}/{}], losses: [Train: {:.5f}, Test: {:.5f}]".format(steps, max_iters, losses['train'], losses['test']))
    
    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

idx = torch.zeros((1, 2), dtype = torch.long, device=device) #== '/n'
max_new_tokens = 300
pred = m.generate(idx, max_new_tokens)

if model_choice == 'Ngram':
    name = f"{model_choice}_{Ngram}gram_{max_iters}"
else:
    name = f"{model_choice}_MI{max_iters}_lr{learning_rate}"
path_dir = path_save_model + name 
os.mkdir(path_dir)
#Save everything in the models file
path = path_dir + "/" + name + ".pt"
torch.save(m, path)
array = np.array([train, test])
np.savetxt(path_dir + "/" + name + "losses.csv", array, delimiter=",")
pred_txt = np.array(decode(pred[0].tolist()))
print(pred_txt.shape)
with open(path_dir+ "/"+ name + "pred.txt", "w") as f:
    f.write(decode(pred[0].tolist()))








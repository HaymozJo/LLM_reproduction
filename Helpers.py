from datasets import load_dataset, get_dataset_split_names, get_dataset_config_names
from transformers import BertTokenizer, XLNetTokenizer
import torch
#Hyperparameters------------------------------
iters = 1000 #progress shown every #iters while encoding phrases
#---------------------------------------------
class DataLoader():

    def __init__(self):
        pass

    def txt_loader(path_input):
        with open(path_input, 'r', encoding='utf-8') as f:
            text = f.read()
        chars = sorted(list(set(text)))
        vocab_size = len(chars)
        return text, chars, vocab_size
    
    #Need to be logged in huggingface
    #trust_remote_code set true, while loading remote code is put. Check remote code first from the datasets
    def hf_loader(self, path, split_bool = False, config_bool = False):
        if split_bool:
            split_q = input("You want to use splits? (y/n)")
            split_bool = True if split_q=='y' else False
            if split_bool:
                split_names = get_dataset_split_names(path, trust_remote_code=True)
                print(f"splits names are : {split_names}")
                split = ""
                while split not in split_names:
                    split = input("choose your split from the list above (only string, no need for "")")
                print(f"valid split: {split} set")
        if config_bool:
            config_q = input("You want to use configs? (y/n)")
            config_bool = True if config_q=='y' else False
            if config_bool:
                config_names = get_dataset_config_names(path, trust_remote_code=True)
                print(f"config names are : {config_names}")
                config = ""
                while config not in config_names:
                    config = input("choose your configs from the list above (only string, no need for "")")
                print(f"valid config: {config} set")

        if split_bool:
            ds = load_dataset(path,config,  split=split) if config_bool else load_dataset(path, split=split)
        else:
            ds = load_dataset(path,config) if config_bool else load_dataset(path)
        return ds
    

    def get_batch(self, split, data):
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))#generate random offsets, start of our different block sequences
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])

        x,y = x.to(self.device), y.to(self.device)
        return x, y


class Encoding():
    def __init__(self, input_tokens = None):
        self.input_tokens = input_tokens
        self.status = 0 #count to deal with the advancements
    
    def IntegerEncoding(self):
        stoi = { ch:i for i,ch in enumerate(self.input_tokens) }
        itos = { i:ch for i,ch in enumerate(self.input_tokens) }
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
        return encode, decode

    def BertToken(self, size_df, progress = True):
        tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
        encode = lambda s: self.encodeWprogress(tokenizer, s, size_df, progress)
        decode = lambda tokens:self.decodeWProgress(tokenizer, tokens, size_df, progress)
        return encode, decode
    
    #Note: need SentencePiece library
    #Include a '_' character before every space to ensure a better understanding in case of non-expected token
    def XLNetToken(self, size_df, progress = True):
        tokenizer = XLNetTokenizer.from_pretrained("xlnet/xlnet-base-cased")
        encode = lambda s: self.encodeWprogress(tokenizer, s, size_df, progress)
        decode = lambda tokens:self.decodeWProgress(tokenizer, tokens, size_df, progress)
        return encode, decode

    #Encode/decode and show advancement of the process
    def encodeWprogress(self, tokenizer, s, total, progress):
        if (self.status%1000 ==0) & (progress):
            print(f"status: {self.status}/{total}", end = "\r", flush=True)
        self.status +=1
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s))
    
    def decodeWProgress(self, tokenizer, tok, total, progress):
        if (self.status%1000 ==0) & (progress):
            print(f"status: {self.status}/{total}")
        self.status += 1
        return  tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(tok))
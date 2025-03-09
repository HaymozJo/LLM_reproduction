import pandas as pd
from datasets import load_dataset, get_dataset_split_names, get_dataset_config_names

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
    def hf_loader(self, path, split_bool = True, config_bool = True):
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
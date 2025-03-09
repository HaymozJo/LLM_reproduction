import pandas as pd
from datasets import load_dataset_builder
from DataLoader import DataLoader
"""
splits = {'train': 'train.csv', 'test': 'test.csv'}
df = pd.read_csv("hf://datasets/Trelis/tiny-shakespeare/" + splits["train"])
"""
path = "Trelis/tiny-shakespeare"

DL = DataLoader()
ds = DL.hf_loader(path)
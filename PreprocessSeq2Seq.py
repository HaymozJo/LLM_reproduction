from Helpers import DataLoader, Encoding
import pandas as pd
import torch
import numpy as np
import re


"""
Preprocess the data for a Sequence to Sequence task (here English to French translation). 
Separated from the train section as loading and transforming all the text into tokens can take some time
"""
#Hyperparameters------------------------------
hf_song_ds = "Nicolas-BZRD/English_French_Songs_Lyrics_Translation_Original"
hf_wmt_gen_ds = "wmt/wmt14"
split = 0.9
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
vocab_size = 30522  # Size of BERT's vocabulary
path_save_dir = "Data/" #save path for the dataframe with input tokens and target tokens
encoding = 'Bert' #Choose encoding in ['Bert', 'XLNet', 'Int']
progress = True #show progress while encoding
#---------------------------------------------
dl = DataLoader()

ds = dl.hf_loader(hf_song_ds)
#Use no as split usage
train_split = ds['train']
#If our dataset is not that big, put it into pandas
df = train_split.to_pandas()
#filter it to only keep english to french songs (takes out ~24 000 songs)
df = df[df['language'] == 'en']
#Keep only relevant information (maybe for future version add the other as context in the encoding)
df = df[['original_version', 'french_version']]

#See exploration to understant the split:
#Note, I took out the count as it was only for a exploration idea
def split_string(str):
    sep_added = re.sub(r'([a-zàâäæáãåāèéêëęėēîïīįíìôōøõóòöœùûüūúÿç])([A-ZÀÂÄÆÁÃÅĀÈÉÊËĘĖĒÎÏĪĮÍÌÔŌØÕÓÒÖŒÙÛÜŪÚŸÇ])', r'\1\n\2', str)
    lines = sep_added.split(sep = '\n')
    lines = [s for s in lines if len(s) >= 10]
    return lines

def split_both(list_origin, list_french, origin, french):
    lines_origin = split_string(origin)
    lines_french = split_string(french)
    if len(lines_origin)==len(lines_french):
        for i in range(len(lines_origin)):
            list_origin.append(lines_origin[i])
            list_french.append(lines_french[i])
    return list_origin, list_french

list_origin, list_french = [], []

for origin, french in zip(df["original_version"], df["french_version"]):
    list_origin, list_french = split_both(list_origin, list_french, origin, french)

#create new df from the lists:
df_split = pd.DataFrame(list(zip(list_origin, list_french)), 
                      columns=["original_version", "french_version"])

print(df_split.info())
#*2 to include french and english~~
size_df = len(df_split.index) *2
#tokenize all the data for our transformers
#Note: French and Enlgish are relatively similar such that using the same tokenizer for both should work well

if encoding == 'Bert':
    encode, decode = Encoding().BertToken(size_df, progress)
elif encoding == 'XLNet':
    encode, decode = Encoding().XLNetToken(size_df, progress)

print("------------English-----------------")
df['input_token'] = df_split['original_version'].apply(encode)
print("------------French------------------")
df['target_token'] = df_split['french_version'].apply(encode)
#Could add a 'Contextual_token' column to add the song context

#Save the input and target tokens into our path_save
df_save = df[['input_token', 'target_token']].apply(np.array)
path_save = path_save_dir + encoding + ".parquet"
df_save.to_parquet(path_save)

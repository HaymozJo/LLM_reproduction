from Helpers import DataLoader, Encoding
import pandas as pd
import torch


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

size_df = len(df.index) *2
#tokenize all the data for our transformers
#Note: French and Enlgish are relatively similar such that using the same tokenizer for both should work well

if encoding == 'Bert':
    encode, decode = Encoding().BertToken(size_df, progress)
elif encoding == 'XLNet':
    encode, decode = Encoding().XLNetToken(size_df, progress)

print("------------English-----------------")
df['input_token'] = df['original_version'].apply(encode)
print("------------French------------------")
df['target_token'] = df['french_version'].apply(encode)
#Could add a 'Contextual_token' column to add the song context

#Save the input and target tokens into our path_save
df_save = df[['input_token', 'target_token']]
path_save = path_save_dir + encoding + ".csv"
df_save.to_csv(path_or_buf=path_save, index=False)
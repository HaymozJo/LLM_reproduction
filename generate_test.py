import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import os
from torch.nn.utils.rnn import pad_sequence 
from AIAYNModel import AIAYNModel
from torchinfo import summary
import numpy as np
from Helpers import Encoding

# Load the trained model
model_path = "Models/Bert_MI50000_lr0.0001_wd0.01/Bert_MI50000_lr0.0001_wd0.01.pt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Model parameters (matching training parameters)
vocab_size = 30522  # BERT vocabulary size
block_size = 32
n_embd = 256  # Changed to match training
n_head = 6
n_layer = 6  # Changed to match training

# Load the model directly
model = torch.load(model_path, map_location=device)
model.eval()  # Set to evaluation mode

# Initialize the encoding helper
encoding = Encoding()
encode, decode = encoding.BertToken(size_df=1, progress=False)  # size_df=1 since we're only processing one sentence

# Function to prepare input
def prepare_input(text):
    # Tokenize the input text
    tokens = encode(text)
    # Add padding if needed
    tokens = encoding.add_padding(tokens, model.block_size)
    # Convert to tensor and move to device
    return torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)  # Add batch dimension

def process_output(generated_tokens):
    # Remove BOS token and everything after EOS token
    tokens = generated_tokens.tolist()
    if model.EOS_TOKEN_ID in tokens:
        tokens = tokens[:tokens.index(model.EOS_TOKEN_ID)]
    if model.BOS_TOKEN_ID in tokens:
        tokens = tokens[tokens.index(model.BOS_TOKEN_ID)+1:]
    # Remove PAD tokens
    tokens = [t for t in tokens if t != model.PAD_TOKEN_ID]
    return tokens

# Example usage
if __name__ == "__main__":
    # Example input text
    test_text = "I love the Sun of Barcelona"
    
    # Prepare input
    input_ids = prepare_input(test_text)
    
    # Generate translation with improved parameters
    generated = model.generate(
        input_ids, 
        max_token=32,
        temperature=0.7,  # Controls randomness (lower = more deterministic)
        top_k=50  # Only consider top 50 tokens at each step
    )
    
    # Process and decode the generated sequence
    processed_tokens = process_output(generated[0])
    generated_text = decode(processed_tokens)
    
    print(f"Input: {test_text}")
    print(f"Generated: {generated_text}")
    
    # Try different temperatures for comparison
    temperatures = [0.5, 0.7, 1.0]
    print("\nGenerations with different temperatures:")
    for temp in temperatures:
        generated = model.generate(input_ids, max_token=32, temperature=temp, top_k=10)
        processed_tokens = process_output(generated[0])
        generated_text = decode(processed_tokens)
        print(f"Temperature {temp}: {generated_text}")


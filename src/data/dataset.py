
# a character level dataset for next-token prediction on a text corpus
import os
import urllib.request
from pathlib import Path
import torch
from torch.utils.data import Dataset

data_dir = Path(__file__).parent.parent / "data"
data_path = data_dir / "tiny_shakespeare.txt"
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

if not data_path.exists():
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset into {data_path}")
    urllib.request.urlretrieve(url, data_path)
    print("download complete.")

class TextDataset(Dataset):
    """
    textdataset: this creates input and target pairs from a text file so a model can learn to predict the next character
    step-by-step approach:
    1. loads the entire text into memory once so disk input/output only happesn at initialization
    2. finds every unique char (our "alphabet"). this hel-s build sthe character vocab
    3. creates mappings:
        stoi: charcater -> integer index for model output
        itos: integer index -> character for decoding

    4. the full yexyt is then encoded as a 1 dimensional tensor of toekn IDs for fast slicing
    5. implements __len__ and __getitem__ to provide pytorch with examples of shape [block_size]
    """

    def __init__ (self, block_size: int):
        #read the raw text file
        text = data_path.read_text(encoding="utf-8")

        #  build the vocab (all unique characters)
        #gets unique characters form the text
        #then these characters are then soprtedn so that the mapping is deterministic
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)

        #creates 2 lookup tables
        #-stoi: (string->index) converts chars to IDs
        #itos: (index->string) for converting IDs back to charcaters

        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}


        #encode the entire text as a one dimensional tensor of integre indices
        #this will enable me to quiclye slice out the training examples in __getitem__
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

        #store the sequnece length (aka block size) for geenerating (input, target) pairs
        self.block_size = block_size


    def __len__(self):
        # Total number of samples is the length of the data minus one block
        # Because at index i, i need i + block_size to be valid for input
        # and i + block_size + 1 to be valid for the target
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        #returns one training example: a pair (x, y) where:
        #x is a sequence of length block_size
        #y is the same sequence shifted ny one character

        #input sequence: iDs from  position idx up to idx + block_size (exclusive)
        x = self.data[idx : idx + self.block_size]

        #target sequence: IDs from position idx+1 up to idx +block_size +1
        #the model will then learn to predict the next character

        y = self.data[idx + 1 : idx + 1 + self.block_size]

        return x, y
    



# a character level dataset for next-token prediction on a text corpus
import urllib.request
from pathlib import Path
import torch
from torch.utils.data import Dataset

default_data_dir = Path(__file__).parent.parent / "data"
default_data_path = default_data_dir / "tiny_shakespeare.txt"
download_url = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)

class TextDataset(Dataset):
    """
    textdataset: this creates input and target pairs from a text file so a model can learn to predict the next character
    step-by-step approach:
    1. loads the entire text into memory once so disk input/output only happesn at initialization
    2. finds every unique char (our "alphabet"). this hel-s build sthe character vocab
    3. creates mappings:
        stoi: charcater -> integer index for model output
        itos: integer index -> character for decoding

    4. the full text is then encoded as a 1 dimensional tensor of toekn IDs for fast slicing
    5. implements __len__ and __getitem__ to provide pytorch with examples of shape [block_size]
    """

    def __init__ (self, block_size: int, file_path: str = None):
    #args:
    #block size:number of tokens per inpout sequence
    #file_path: the optional path to a text file. If none, it uses the default shakespeare file
        
        #read the raw text file
        self.data_path = Path(file_path) if file_path else default_data_path

        #download if the file is missing
        if not self.data_path.exists():
            print(f"Downloading dataset to {self.data_path}...")
            urllib.request.urlretrieve(download_url, self.data_path)
            print("download is complete!")

        #get the full text into memory
        text = self.data_path.read_text(encoding="utf-8")


        #construct the sorted list of unique vocab
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)

        #crete the nappings for each char to index and back
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
    


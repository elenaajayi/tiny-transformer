
# a character level dataset for next-token prediction on a text corpus
from pathlib import Path
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    """
    textdataset: this creates input and target pairs from a text file so a model can learn to predict the next character
    step-by-step approach:
    loads the entire text into memory
    finds every unique char (our "alphabet").
    converts chars to numbers and back (therefore teh computer can work with them).
    """
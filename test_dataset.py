#smoke test for the etxt dataset: this will test if the datasst was actually downloaded, preprocessed, and logic has been indexed

from src.data.dataset import TextDataset

# start the dataset with a small block size for easy inspection
#block_size = 10, the input, x, will consist of 10 characters
#y will be the next 10 characters (shifted by one).

ds = TextDataset(block_size = 10)

#checking if thv dataste loaded properly
#len(ds) this should be = total_chars - block_size
#this will tell me how many (input, target) pairs I can generate
print("Dataset length:", len(ds))

#ds.vocab_size = unique chaarcters in the corpus
#tells me the output dimension of the model's head

print("Vocab size:", ds.vocab_size)


#I will inspect the first example here (x,y)
#x : tensor of shape (block_size) - this contains the interger IDS for each character
#y: tensor of shape (block_size), this represents the next sequence of characters

x,y = ds[0]

print("First Input, as IDs:", x)
print("First Target, as IDs:", y)

# decoding ids to check how visually correct they are

#ds.itos maps each integer ID back to its original character
decoded_x = ''.join(ds.itos[int(1)] for i in x)
decoded_y = ''.join(ds.itos[int(1)] for i in y)

print("decoded input:", decoded_x)
print("Decoded target:", decoded_y)

#if the decoded output matches the subsequences opf the text i expect, that means, the way i downloaded, built the vocab, and sliced the logic are working correctly :D
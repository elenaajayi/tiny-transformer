import torch
from src.model.transformer_block import TransformerBlock

#setting the params
batch_size = 2        # number of input sequences in one batch (like 2 sentences)
seq_len = 5            # number of tokens per sequence (e.g., each "sentence" has 5 tokens)
embed_dim = 32         # each token is represented by a 32-dimensional vector
num_heads = 4          # how many attention "focuses" the block uses at once
ff_hidden_dim = 64     # internal size of the feedforward layer (hidden "thinking space")
dropout = 0.1          # dropout adds randomness to prevent overfitting

#dummy data; what the transformer would receieve during training
#batch embedded tokens, shaeped as [batch_size, sequence_length, embedding_dim]
dummy_input = torch.randn(batch_size, seq_len, embed_dim)

#transformerblock with test params
#one full block, including self-attention, feedforward, layernorms, and dropout
block = TransformerBlock(embed_dim, num_heads, ff_hidden_dim, dropout)

#passing the dummy data into the tranformer block to check whether the blocjk can handle the input shape, run computattions, and return a valid output (does it work)?
output = block(dummy_input)

#print the input and output shaopes
# should match excatly -- the transformer blocks shoiuld keep the same shape, only the contents should be changed
print("input Shape:" , dummy_input.shape)
print("output shape:", output.shape)
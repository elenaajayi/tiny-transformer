import torch
import torch.nn as nn


# Transformer block that will be stacked in the mini transformer
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
        # args:
        # embed_dim: dimensionality of token embeddings and internal representation;
        #            how many numbers we use to represent each token
        # num_heads: how many separate attention focuses the model uses at once
        #            (how many angles is a sentence looked at from?)
        # ff_hidden_dim: controls how "wide" the thinking is between attention layers
        # dropout: randomly drops some values from the model during training to prevent overfitting;
        #          adds noise, making the model less reliant on specific patterns
            super().__init__()

            #multi-head self-attention: lets the model focus on different parts of the sequence at once
            self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

            #first layer normalization(after attn): helps stabilize training and improve the flow of gradients
            self.norm1 = nn.LayerNorm(embed_dim)

            #feedforward network: a mini two layer neural network that is applied to each token
            self.ff = nn.Sequential(
                  nn.Linear(embed_dim, ff_hidden_dim), #the first layer that expands thinking
                  nn.ReLU(), #adds non-linearity 
                  nn.Linear(ff_hidden_dim, embed_dim) #gets it back to its orig size
            )

            #second layer normalization: this is applied after the feed forward network
            self.norm2 = nn.LayerNorm(embed_dim)

            #dropupout layer: adds randomness to prevent overfitting
            self.dropout = nn.Dropout(dropout)
    def forward(self, x):
          # x: input tensor of shape [batch_size, seq_len, embed_dim]

           # self-attention step — the model checks how each token relates to every other token
        attn_out, _ = self.attn(x, x, x)  # self-attention means we use the same x for query, key, and value

            # add & norm: combine input and attention output, then normalize
        x = self.norm1(x + self.dropout(attn_out))

            #feedforward step -- deeper thinking at eachposition
        ff_out = self.ff(x)

            # another add & norm: combine the inout and feedforward boutput, then normalize
        x = self.norm2(x + self.dropout(ff_out))

        return x
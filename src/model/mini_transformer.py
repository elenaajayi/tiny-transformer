import torch
import torch.nn as nn
from src.model.transformer_block import TransformerBlock

class MiniTransformer(nn.Module):
    def __init__(
        # args:
        self,
        vocab_size, # total number of distinct tokens (like the size of your puzzle box)
        embed_dim,      # how many “features” describe each token—in other words, the length of each token’s vector
        num_heads,      # how many separate attention “eyes” the model uses at once to look at a sentence from different angles
        ff_hidden_dim,  # size of the inner layer in the little per-token MLP—controls how much extra “thinking power” each token has
        num_layers,     # how many of these attention+MLP blocks we stack; more layers = deeper reasoning 
        max_seq_len,    # the longest sentence (in tokens) the model can process at once—its “memory window” length
        dropout=0.1,    # fraction of features we randomly ignore during training so the model doesn’t latch onto spurious patterns
    ):
    
        super().__init__()


        # the tojken embedding layer: thus will transform token IDs into dense vectors
        #ex. lookup table converting words into their vector representations

        self.token_emb = nn.Embedding(vocab_size, embed_dim)

        #positional embedding layer: learns a vector for each position ion the sequence
        #modle then gets a sense of the word order, like numberung each word's place in the sentence
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)


        #stack of transforer block: each block applies sekf-attention and a small feed-forward network
        #when there are more layers, there is more abstract understanding of the etxt 
        self.block = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        ##Final layer normalization: helps keep the represnetations more stable before the output layer
        #LayerNorm: thsi standardizes the activations across the embedding dimension
        self.ln_f = nn.LayerNorm(embed_dim)

        #output head: this is a linear layer that will map the final embeddings to vocab-sized logits
        #these logits will be use tp calculate the probabilities of the next token

        self.head = nn.Linear(embed_dim, vocab_size)


        def forward(self, x):
            #forward pas to the MiniTransformer.
            #args:
            #x: longTensor of shape [batch_size, seq_len], containing token IDs

            #returns:
            #logits: float tensor of the spae [batch-size, seq_len, vocab-size],
            #raw scores of the next-token prediction at each position

            batch_size, seq_len = x.size()

            #create a tensor of positions [0,1,2,..., seq_len-1], shape [1, seq_len]
            #when i expand this for each batch, it will give a positional index fpor each token (word)

            positions = torch.arrange(seq_len, device=x.device).unsqueeze(0)

            #llokup[ embeddings and add: shape becomes [batch_size, seq_len, embed_dim]
            #summing token + position embeddingswill merge the word identity with its location info
            x = self.token_emb(x) + self.pos_emb(positions)

            #pass the embeddings through each TransformerBlock in sequence
            #each block refines the token vectors by mixing contextual information through attention

            for block in self.blocks:
                x = block(x)

            #now, lest normalize the final representations before we project it to the logits

            x = self.ln_f(x)
            
            #project to vacab dimensions: [batcgh_size, seq_len, vocab_size]
            # these logits indicate the model's raw predictions for each posiible next token
            logits = self.head(x)
            return logits
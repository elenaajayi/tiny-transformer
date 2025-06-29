
"""
src/train_loop/train.py

Runs the full training process for our MiniTransformer model:
1) Load settings (like batch size, learning rate) from a YAML file.
2) Prepare text data and split it into batches.
3) Build the transformer, set up the loss function, and choose an optimizer.
4) For each batch:
   - Clear old gradients.
   - Do a forward pass (model makes predictions).
   - Compute the loss (how far off predictions are).
   - Backward pass (figure out how to adjust weights).
   - Update weights (optimize to reduce loss).
   - Keep track of loss to see how training is going.
5) Save the final model to disk.
"""

import sys
print("running train.py from:", __file__)
print("   Python sys.path[0] is:", sys.path[0])


import yaml  # Used to load hyperparameters from a YAML config file. Install with: pip install pyyaml
from pathlib import Path # Simplifies file path manipulations
# This allows easy parsing of YAML files into Python dictionaries,
# making it simple to update and test different model training settings without changing the code.

import torch  # PyTorch library for tensors and GPU support
import torch.nn as nn  # Neural network components (layers, loss functions)
from torch.utils.data import DataLoader  # Manages batching and shuffling of data


from src.model.mini_transformer import MiniTransformer  # Your custom GPT-style model
from src.data.dataset import TextDataset  # Dataset that auto-downloads, tokenizes, and slices text

#going to use a config file to make sure the code is clean and the parameneters are not a hassle to adjust
# Get absolute path to config.yaml (most reliable method)
project_root = Path.cwd()
config_file = project_root / 'config' / 'config.yaml'
config = yaml.safe_load(config_file.read_text())

#get our hyperparams
batch_size = config['batch_size'] # number of sequences per training batch
epochs = config ['epochs'] # how many times am i going to loop over the dataset
lr = config['lr'] # the step size for the optimizer
seq_len = config ['seq_len'] #what is the  number of token per each input sequence
#use the GPU for faster training =, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() and config['device']=='cuda' else 'cpu'

#resoving data_path from config

data_path = Path(config['data_path'])
if not data_path.is_absolute():
    data_path = project_root / data_path
print(f"Loading data from {data_path}")
data_path.parent.mkdir(parents=True, exist_ok=True)
#now lets prepare the data
#TextDataset - handles the file donwload, vocab creation, and example slicing
#DataLoader will batch examples and shuffle training robustness



print(f"loading dataset with block size={seq_len}, path={config['data_path']}")
dataset = TextDataset(
    block_size=seq_len,
    file_path = config['data_path'])

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#build the model, the loss, and the optimize
#miniTransfomer: thsi will stack token embeddings with transformer blocks
#crossentropyloss: this will calculate the loss for the next token prediction
#adamw: this is the daaptoive optimizer for the transformer weight updates


transformer_model = MiniTransformer(
    vocab_size = dataset.vocab_size,
    embed_dim = config.get('embed_dim', 128),
    num_heads = config.get('num_heads', 4),
    ff_hidden_dim = config.get('ff_hidden_dim', 512),
    num_layers = config.get('num_layers', 4),
    max_seq_len = seq_len,
    dropout = config.get('dropout', 0.1)
).to(device) # move the transformer to gpu is vaailable
criterion = nn.CrossEntropyLoss() # loss for classification over the vocabulary
optimizer = torch.optim.AdamW(transformer_model.parameters(), lr=lr) #optimizer with the weight decay

#training loop
# for each batch, i will perform forward pass, compute the loss, backpropagate, and then update the weights
print(f"training on {device} for {epochs} epochs...")
for epoch in range(1, epochs + 1):
    transformer_model.train() # set the training mode
    running_loss = 0.0

for idx, (x, y) in enumerate(dataloader, start=1):
    #move data to GPU or CPU
    x,y = x.to(device), y.to(device)

    #clear the prev gradients
    optimizer.zero_grad()

    #forward pass: this makes our transfomer_model predictions (logits)

    logits = transformer_model(x) #s shape: [batch, seq_len, vocab_size]

    #compute loss: this will compare the predictions to actual next tokens
    #flatten tensors: [batch*seq_len, vocab_size] and [batch*seq_len]

    loss = criterion(logits.view(-1, dataset.vocab_size), y.view(-1))

    #backward pass:calculate how to change the weights to reduce the loss
    loss.backward()

    #update the wights: apply the gradient step
    optimizer.step()

    #accumulate loss for monitoring
    running_loss += loss.item()

    # every 100 batches, print the average loss so far
    if idx % 100 == 0:
        print(f"Epoch {epoch} Batch {idx}/{len(dataloader)} Loss {running_loss/idx:.4f}")
    
    print(f"Epoch {epoch} complete. Avg Loss: {running_loss/len(dataloader):.4f}\n")
    


#save the trained model
output_dir = project_root / 'outputs' / 'checkpoints'
output_dir.mkdir(parents=True, exist_ok=True)
torch.save(transformer.state_dict(), output_dir / 'mini_transformer.pt')




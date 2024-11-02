# dataset
import torch
from torch import nn
from torch.nn import functional as F

# Configuration / hyperparameters
batch_size = 4 # batch size: // 4 for testing
block_size = 8 # block size: // 8 for testing
embedding_size = 3 # embedding size: // 3 for testing
head_size = 3 # head size: // 3 for testing
dropout_rate = 0.1 # dropout rate: // 0.1 for testing
eval_interval = 100 # evaluation interval: // 100 for testing
num_heads = 1 # number of heads: // 1 for testing
max_iters = 1000 # maximum iterations: // 1000 for testing
learning_rate = 2e-4 # learning rate: // 2e-4 for testing
device = "cuda" if torch.cuda.is_available() else "cpu" # device agnostic code cuda or cpu
eval_iters = 10 # evaluation iterations: // 10 for testing
n_layer = 1 # number of layers: // 1 for testing

with open("./input.txt", "r", encoding="utf-8") as file:
    text = file.read()
    
chars = sorted(list(set(text))) # all the unique character of our vocabulary
vocab_size = len(chars) # 65 vocab size for us

stoi = { ch:i for i, ch in enumerate(chars) } # char to index
itos = { i:ch for i, ch in enumerate(chars) } # index to char

def encode_text(text: str) -> list[int]:
    """Convert text to number based on vocabulary"""
    return [stoi[ch] for ch in text]

def decode_text(numbers: list[int]) -> str:
    """Convert numbers to text based on vocabulary"""
    return "".join([itos[i] for i in numbers])

# convert text to numbers
data = torch.tensor(encode_text(text), dtype=torch.long)

# split the data
split_size = int(0.9 * len(data)) # 90% train, 10% test

train_data = data[:split_size]
val_data = data[split_size:]

# get the batch of data
def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Get the random batch of data from the train or validation data."""
    data = train_data if split == "train" else val_data
    random_start_ix = torch.randint(len(data) - block_size, (batch_size, )) # get random batch_size integer from 0 to len(data) - block_size
    x = torch.stack([data[i:i+block_size] for i in random_start_ix]) # get the block_size int from random_start_ix as training data
    y = torch.stack([data[i+1:i+block_size+1] for i in random_start_ix]) # get the block_size int from random_start_ix shifted by 1 as label data
    return x, y

# attention mechanism (scaled dot-product attention)
class Head(torch.nn.Module):
    """Single attention head with scaled dot-product attention."""
    def __init__(self):
        super().__init__()
        self.key = nn.Linear(embedding_size, head_size, bias=False)
        self.query = nn.Linear(embedding_size, head_size, bias=False)
        self.value = nn.Linear(embedding_size, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        # B, T, C dot B, C, T -> B, T, T
        attention_weight = torch.matmul(q, k.transpose(-2, -1)) * C ** -0.5 # scaled dot-product attention
        attention_weight = attention_weight.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # mask the future information
        attention_weight = F.softmax(attention_weight, dim=-1)
        attention_weight = self.dropout(attention_weight)
        
        v = self.value(x)
        attention_score = torch.matmul(attention_weight, v) # B, T, T dot B, T, C -> B, T, C
        
        return attention_score
    
class FeedForward(torch.nn.Module):
    """Feed forward network with ReLU activation."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_size, embedding_size * 4),
            nn.ReLU(),
            nn.Linear(embedding_size * 4, embedding_size),
            nn.Dropout(dropout_rate)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head() for _ in range(num_heads)]
        )
        self.proj = nn.Linear(embedding_size, embedding_size)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        multihead_output = torch.cat(
            [head(x) for head in self.heads]
        )
        multihead_output = self.proj(multihead_output)
        multihead_output = self.dropout(multihead_output)
        
        return multihead_output
    
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = embedding_size // num_heads
        self.attention = MultiHeadAttention()
        self.ffwd = FeedForward()
        self.layernorm1 = nn.LayerNorm(embedding_size)
        self.layernorm2 = nn.LayerNorm(embedding_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(x)
        x = x + self.ffwd(x)
        x = x + self.attention(self.layernorm1(x))
        x = x + self.ffwd(self.layernorm2(x))
        
        return x
    
# metric
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss
            out[split] = losses.mean()
        model.train()
    return out

# our model
class NanoGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_token = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding_table = nn.Embedding(block_size, embedding_size)
        self.blocks =  nn.Sequential(
            *[Block() for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(embedding_size)
        self.lm_head = nn.Linear(in_features=embedding_size, out_features=vocab_size)
        
    def forward(self, idx, targets=None):
        """Forward pass of the model.
        
        Args:
            idx (torch.Tensor): Input tensor of shape (B, T).
            targets (torch.Tensor): Target tensor of shape (B, T).
        """
        B, T = idx.shape
        token_embedding = self.token_embedding_token(idx)
        position_embedding = self.position_embedding_table(torch.arange(T, device=device))
        x = token_embedding + position_embedding
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        """Generate new tokens of length max_new_tokens.
        Args:
            idx (torch.Tensor): Input tensor of shape (B, T).
            max_new_tokens (int): Maximum number of tokens to generate.
        Returns:
            torch.Tensor: Tensor of shape (B, T + max_new_tokens) containing the generated tokens.
        """
        for _ in range(max_new_tokens):
            idx_condition = idx[:, -block_size:] # only get the last block_size tokens
            logits, loss = self(idx_condition)
            
            logits = logits[:, -1, :] # B, C get the only for the last token
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # B, 1
            idx = torch.cat((idx, idx_next), dim=1) # B, T + 1
        
        return idx
    
# using model
model = NanoGPT().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train the model
for epoch in range(max_iters):
    # keep track of the loss and accuracy for both the training and validation set for each epoch
    if epoch % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"Epoch {epoch} | Train Loss: {losses['train']} | Val Loss: {losses['val']}")
        
    xb, yb = get_batch("train")
    
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
# generate text
context = torch.zeros((1,1), dtype=torch.long, device=device)
generated_token = model.generate(context, 100)[0].tolist()
final_output_text = decode_text(generated_token)
print(final_output_text)
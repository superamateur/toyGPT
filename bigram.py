# implement a bigram language model
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

torch.manual_seed(1337)


@dataclass
class TrainConfig:
    batch_size = 32
    block_size = 8
    max_iter = 5000
    eval_interval = 500
    lr = 1.e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 200
    n_embd = 32

cfg = TrainConfig()

class DataFactory:
    def __init__(self, input_txt):
        with open(input_txt, 'r', encoding='utf-8') as f:
            self.text = f.read()
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        stoi = { ch: i for i, ch in enumerate(self.chars)}
        itos = { i: ch for i, ch in enumerate(self.chars)}
        self.encode = lambda s: [stoi[c] for c in s] # input: string, output: list of integers
        self.decode = lambda l: ''.join([itos[i] for i in l]) # input: list of integers, output: string
        self.data = torch.tensor(self.encode(self.text), dtype=torch.long)
        n = int(0.9 * len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]
    
    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        # randomly select the starting index of text blocks
        ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size, ))
        x = torch.stack([data[i: i + cfg.block_size] for i in ix])
        y = torch.stack([data[i + 1: i + cfg.block_size + 1] for i in ix])
        x, y = x.to(device=cfg.device), y.to(device=cfg.device)
        return x, y

data_factory = DataFactory(input_txt='input.txt')

@torch.no_grad()
def estimate_loss(net):
    out = {}
    net.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            X, Y = data_factory.get_batch(split=split)
            logits, loss = net(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    net.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(cfg.block_size, cfg.block_size)))
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q  @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    '''
    Multiple heads of self attention in parallel
    '''
    def __init__(self, num_heads, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    
    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity"""
    def __init__(self, n_embd) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd), nn.ReLU()
        )
    
    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head) -> None:
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
    
    def forward(self, x):
        return self.net(x)
    
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size) -> None:
        super().__init__()
        self.token_embeding_table = nn.Embedding(vocab_size, cfg.n_embd)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)

        self.sa_heads = MultiHeadAttention(4, cfg.n_embd // 4)
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size)

        self.ffwd = FeedForward(cfg.n_embd)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_embd = self.token_embeding_table(idx) # shape: (B, T, C) = (batch, time, channel) = (batch_size, block_size, n_embd)
        pos_embd = self.position_embedding_table(torch.arange(T, device=cfg.device)) # (T, C=n_embd)
        x = tok_embd + pos_embd

        x = self.sa_heads(x)
        x = self.ffwd(x)
        logits = self.lm_head(x) # (batch_size, block_size, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(-1) # B *T 
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        '''
        Takes in an input of shape (B,T), generate output of shape (B, T + max_new_tokens) 
        '''
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -cfg.block_size:]
            logits, _ = self.forward(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # (B, C)
            # convert logits to probabilities
            prob = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(prob, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T + 1)
        return idx

m = BigramLanguageModel(vocab_size=data_factory.vocab_size).to(device=cfg.device)

optimizer = torch.optim.AdamW(m.parameters(), lr=cfg.lr)

for iter in range(cfg.max_iter):
    if iter % cfg.eval_interval == 0:
        losses = estimate_loss(m)
        print(f"step: {iter}, train loss: {losses['train']}, val loss: {losses['val']}")

    xb, yb = data_factory.get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=cfg.device)
outputs = m.generate(context, max_new_tokens=100)
print(data_factory.decode(outputs[0].tolist()))
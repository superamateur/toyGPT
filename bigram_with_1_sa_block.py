import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl


class AttentionHead(nn.Module):
    def __init__(self, head_size, n_embd, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

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

class FeedForward(nn.Module):
    def __init__(self, n_embd) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), 
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # similar to projection layer in the multihead attention
        )
    
    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    '''
    Multiple heads of self attention in parallel
    '''
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout) -> None:
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
    
class AttentionBlock(nn.Module):
    def __init__(self, n_embd, block_size) -> None:
        super().__init__()
        head_size = n_embd
        self.sa = AttentionHead(head_size, n_embd, block_size)
        self.ffwd = FeedForward(n_embd)
        self.layer_norm1 = nn.LayerNorm(n_embd)
        self.layer_norm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.layer_norm1(x))
        x = x + self.ffwd(self.layer_norm2(x))
        return x
        
class BigramLanguageModel1SA(pl.LightningModule):
    def __init__(self, vocab_size, cfg) -> None:
        super().__init__()
        self.token_embeding_table = nn.Embedding(vocab_size, cfg.n_embd)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)

        self.atten_block = AttentionBlock(cfg.n_embd, cfg.block_size)
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size)
        self.register_buffer('block_position', torch.arange(cfg.block_size))
        self.cfg = cfg

    
    def forward(self, idx):
        _, T = idx.shape

        tok_embd = self.token_embeding_table(idx)
        pos_embd = self.position_embedding_table(self.block_position)
        x = tok_embd + pos_embd[:T, :]
        x = self.atten_block(x)
        logits = self.lm_head(x)
        return logits
    
    def generate(self, idx, max_new_tokens):
        '''
        Takes in an input of shape (B,T), generate output of shape (B, T + max_new_tokens) 
        '''
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits = self.forward(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # (B, C)
            # convert logits to probabilities
            prob = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(prob, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T + 1)
        return idx
    
    def calc_loss(self, xb, yb):
        logits = self(xb)
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        yb = yb.view(-1) # B *T 
        loss = F.cross_entropy(logits, yb)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        xb, yb = batch[0]
        loss = self.calc_loss(xb, yb)
        
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        xb, yb = batch
        loss = self.calc_loss(xb, yb)
        
        # Logging to TensorBoard by default
        self.log("val_loss", loss)
        return loss

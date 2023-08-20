import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl


class BigramLanguageModelBase(pl.LightningModule):
    def __init__(self, vocab_size, cfg) -> None:
        super().__init__()
        self.token_embeding_table = nn.Embedding(vocab_size, cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size)
        self.cfg = cfg

    
    def forward(self, idx):
        B, T = idx.shape

        tok_embd = self.token_embeding_table(idx) # shape: (B, T, C) = (batch, time, channel) = (batch_size, block_size, n_embd)
        logits = self.lm_head(tok_embd) # (batch_size, block_size, vocab_size)
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

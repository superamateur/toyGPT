import torch
from torch.utils.data import Dataset

class DataFactory(Dataset):
    def __init__(self, cfg):
        with open(cfg.input, 'r', encoding='utf-8') as f:
            self.text = f.read()
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        stoi = { ch: i for i, ch in enumerate(self.chars)}
        itos = { i: ch for i, ch in enumerate(self.chars)}
        self.encode = lambda s: [stoi[c] for c in s] # input: string, output: list of integers
        self.decode = lambda l: ''.join([itos[i] for i in l]) # input: list of integers, output: string
        self.data = torch.tensor(self.encode(self.text), dtype=torch.long)
        self.block_size = cfg.block_size
        self.length = len(self.data) - self.block_size
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        x = self.data[index: index + self.block_size] # context
        y = self.data[index + 1: index + self.block_size + 1] # target
        return x, y
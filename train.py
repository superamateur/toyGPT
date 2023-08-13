import yaml
import click
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from data_loader import DataFactory
from model import BigramLanguageModel

torch.manual_seed(1337)

@dataclass
class RuntimeConfig:
    input: str
    batch_size: int
    block_size: int
    device: str
    max_iter: int
    eval_interval: int
    lr: float
    eval_iters: int
    n_embd: int
    n_head: int
    n_layer: int
    dropout: float
    checkpoint_path: str

def train_fn(run_cfg):
    # prepare datasets
    dataset = DataFactory(cfg=run_cfg)
    train_dataset, val_dataset = random_split(dataset, [0.9, 0.1])

    train_loader = DataLoader(train_dataset, batch_size=run_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=run_cfg.batch_size, shuffle=False)

    vocab_size = dataset.vocab_size
    model = BigramLanguageModel(vocab_size=vocab_size, cfg=run_cfg)

    # config checkpoint callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=None,
        filename='{epoch}-{step}-{train_loss:.2f}',
        save_last=True,
        monitor='train_loss',
        every_n_train_steps=1000,
        save_top_k=-1 # don't ovewrite checkpoints
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        max_steps=run_cfg.max_iter,
        val_check_interval=run_cfg.eval_interval,
        limit_val_batches=run_cfg.eval_iters,
        callbacks=[checkpoint_callback])
    
    trainer.fit(model=model, train_dataloaders=[train_loader], val_dataloaders=[val_loader])


def test_fn(run_cfg):
    data_factory = DataFactory(cfg=run_cfg)
    vocab_size = data_factory.vocab_size
    model = BigramLanguageModel(vocab_size=vocab_size, cfg=run_cfg)
    model.to("cuda")
    checkpoint = torch.load(run_cfg.checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    # context = torch.zeros((1, 1), dtype=torch.long, device="cuda")
    context = torch.tensor(data_factory.encode("I ain't "), dtype=torch.long, device="cuda").view(1, -1)
    outputs = model.generate(context, max_new_tokens=1000)
    
    print(data_factory.decode(outputs[0].tolist()))


@click.command()
@click.option('--config-file', type=str, required=True, help='Configuration YAML file', default='runtime_config.yaml')
@click.option('--mode', type=click.Choice(['train', 'test']), required=True, help='Run mode', default='train')
def main(config_file: str, mode: str):
    # Load config from yaml file
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    run_cfg = RuntimeConfig(**config)
    if mode == 'train':
        train_fn(run_cfg=run_cfg)
    else:
        test_fn(run_cfg=run_cfg)


if __name__ == '__main__':
    main()
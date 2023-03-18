import yaml
import argparse
from diffusion import Diffusion
import pytorch_lightning as pl
from ema import EMA, EMAModelCheckpoint
from torch.utils.data import DataLoader
from data import get_dataset
from pytorch_lightning.strategies.ddp import DDPStrategy
import click

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

@click.command()
@click.option('--cfg', default='config.yml', help='Configuration File')
def main(cfg):
    with open(cfg, "r") as f:
        cfg = yaml.safe_load(f)
        cfg = dict2namespace(cfg)

    ckpt_callback = EMAModelCheckpoint(save_top_k=5, monitor="val_loss", save_last=True, filename='{epoch}-{val_loss:.4f}', every_n_train_steps=None,)
    ema_callback = EMA(decay=cfg.model.ema_rate)
    callbacks = [ckpt_callback, ema_callback]

    model = Diffusion(cfg)

    train_dataloader = DataLoader(
        get_dataset(cfg.data.name, train=True), 
        batch_size=cfg.training.batch_size, 
        shuffle=True, 
        num_workers=cfg.data.num_workers, 
        pin_memory=True, 
        persistent_workers=True,
    )

    val_dataloader = DataLoader(
        get_dataset(cfg.data.name, train=False), 
        batch_size=cfg.testing.batch_size, 
        shuffle=False, 
        num_workers=cfg.data.num_workers, 
        pin_memory=True, 
        persistent_workers=True,
    )


    trainer = pl.Trainer(
        callbacks=callbacks,
        precision=cfg.training.precision,
        max_steps=cfg.training.max_steps,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        accelerator="gpu", 
        devices=[0,1],
        # limit_val_batches=1,
        gradient_clip_val=cfg.optim.grad_clip,
        benchmark=True,
        strategy = DDPStrategy(find_unused_parameters=False),
    )

    # Train
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

if __name__ == '__main__':
    main()
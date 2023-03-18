import pytorch_lightning as pl
from networks import VEPrecond, VPPrecond, EDMPrecond, CTPrecond
from loss import VELoss, VPLoss, EDMLoss, CTLoss
from torch import optim
import numpy as np
import torch
from sampler import multistep_consistency_sampling
from torchvision.utils import make_grid, save_image
import copy
from torchmetrics.image.inception import InceptionScore
from sampler import multistep_consistency_sampling


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
    
class Diffusion(pl.LightningModule):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        if cfg.diffusion.preconditioning == 'vp':
            self.loss_fn = VPLoss()
            self.net = VPPrecond(self.cfg)
        elif cfg.diffusion.preconditioning == 've':
            self.loss_fn = VELoss()
            self.net = VEPrecond(self.cfg)
        elif cfg.diffusion.preconditioning == 'edm':
            self.loss_fn = EDMLoss()
            self.net = EDMPrecond(self.cfg)
        elif cfg.diffusion.preconditioning == 'ct':
            self.loss_fn = CTLoss(self.cfg)
            self.net = CTPrecond(self.cfg, use_fp16=self.cfg.training.precision == 16)
            self.net_ema = CTPrecond(self.cfg, use_fp16=self.cfg.training.precision == 16) # no_grad or not ???
            for param in self.net_ema.parameters():
                param.requires_grad = False
            self.net_ema.load_state_dict(copy.deepcopy(self.net.state_dict()))
        else:
            raise ValueError(f'Preconditioning {cfg.diffusion.preconditioning} does not exist')
        if self.cfg.testing.calc_inception:
            self.inception = InceptionScore()
            

    def training_step(self, batch, _):
        images, _ = batch
        loss = self.loss_fn(net=self.net, net_ema=self.net_ema, images=images, k=self.global_step).mean()

        with torch.no_grad():
            mu = self.net.mu(self.global_step)
            # update \theta_{-}
            for p, ema_p in zip(self.net.parameters(), self.net_ema.parameters()):
                ema_p.mul_(mu).add_(p, alpha=1 - mu)

        self.log("train_loss", loss, on_step=True)
        return {'loss': loss}

    def validation_step(self, batch, _):
        images, _ = batch
        loss = self.loss_fn(net=self.net, net_ema=self.net_ema, images=images, k=self.global_step).mean()
        self.log("val_loss", loss)
        
        if self.cfg.testing.calc_inception:
            latents = torch.randn(images.shape[0], self.cfg.data.img_channels, self.cfg.data.img_resolution, self.cfg.data.img_resolution).cuda() 
            xh = multistep_consistency_sampling(self.net, latents=latents, t_steps=[80])
            xh = ((xh + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            self.inception.update(xh)

        return {'loss': loss}
    
    
    def validation_epoch_end(self, outputs):
        latents = torch.randn(self.cfg.testing.samples, self.cfg.data.img_channels, self.cfg.data.img_resolution, self.cfg.data.img_resolution).to(self.device) 
        name = self.cfg.data.name
        xh = multistep_consistency_sampling(self.net_ema, latents=latents, t_steps=[80])
        xh = (xh * 0.5 + 0.5).clamp(0, 1)
        grid = make_grid(xh, nrow=4)
        save_image(grid, f"samples/ct_{name}_sample_1step_{self.global_step}.png")
        if self.cfg.testing.calc_inception:
            iscore = self.inception.compute()[0]
            self.log('iscore', iscore)
        return super().validation_epoch_end(outputs)
    
    def configure_optimizers(self):
        cfg = self.cfg.optim
        if cfg.optimizer == 'radam':
            optimizer = optim.RAdam(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas, 
                            eps=cfg.eps)
        elif cfg.optimizer == 'adam':
            optimizer = optim.Adam(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas, 
                            eps=cfg.eps)
        elif cfg.optimizer == 'adamw':
            optimizer = optim.AdamW(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas, 
                            eps=cfg.eps)
        elif cfg.optimizer == 'rmsprop':
            optimizer = optim.RMSprop(self.net.parameters(), lr=cfg.lr)
        elif cfg.optimizer == 'sgd':
            optimizer = optim.SGD(self.net.parameters(), lr=cfg.lr)

        lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=self.cfg.training.warmup_epochs, max_iters=self.cfg.training.max_epochs)

        return {
        "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
            },
        }
    
    
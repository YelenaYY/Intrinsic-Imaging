# train_composer.py
import os
import sys
from pathlib import Path
from datetime import datetime
import csv

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm

from models import Decomposer, NeuralShader
from datasets import IntrinsicDataset
from utils.checkpoint import find_lastest_checkpoint

def normalize_normals(n):
    mag = n.pow(2).sum(1, keepdim=True).sqrt().clamp_min(1e-6)
    return n / mag

def masked_l1(pred, target, mask_bool):
    m = mask_bool.expand_as(pred)
    num = m.sum().clamp_min(1)
    return (pred[m] - target[m]).abs().sum() / num

class ComposerTrainer:
    def __init__(self, config):
        self.device = config["train"]["device"]
        print(f"Using device: {self.device}")

        # config
        ccfg = config["train"]["composer"]
        epochs = ccfg["epochs"]
        learning_rate = ccfg["learning_rate"]
        log_folder = ccfg["log_folder"]
        checkpoints_folder = ccfg["checkpoints_folder"]
        train_datasets = ccfg["train_datasets"]
        validate_datasets = ccfg["validate_datasets"]
        light_array = ccfg["light_array"]
        load_latest_checkpoint = config["train"]["load_latest_checkpoint"]
        batch_size = ccfg.get("batch_size", 4)
        lights_dim = ccfg.get("lights_dim", 4)
        expand_dim = ccfg.get("expand_dim", 8)
        freeze_shader = ccfg.get("freeze_shader", True)

        # data
        train_dataset = IntrinsicDataset(train_datasets, light_array)
        validate_dataset = IntrinsicDataset(validate_datasets, light_array)
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Val   dataset size: {len(validate_dataset)}")

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True, pin_memory=True)
        self.validate_loader = DataLoader(validate_dataset, batch_size=batch_size, num_workers=2, shuffle=False, pin_memory=True)

        # models
        self.decomposer = Decomposer(lights_dim=lights_dim).to(self.device)
        self.shader = NeuralShader(lights_dim=lights_dim, expand_dim=expand_dim).to(self.device)

        # Optionally load latest checkpoints (separately for both nets)
        if load_latest_checkpoint:
            latest_dec, dec_num = find_lastest_checkpoint(os.path.join(checkpoints_folder, "decomposer"))
            if latest_dec:
                print(f"Loading decomposer checkpoint: {latest_dec}")
                self.decomposer.load_state_dict(torch.load(latest_dec, map_location=self.device))
            latest_sh, sh_num = find_lastest_checkpoint(os.path.join(checkpoints_folder, "shader"))
            if latest_sh:
                print(f"Loading shader checkpoint: {latest_sh}")
                self.shader.load_state_dict(torch.load(latest_sh, map_location=self.device))

        # Freeze shader if desired (paper-style transfer)
        if freeze_shader:
            for p in self.shader.parameters():
                p.requires_grad = False

        # optimizer over trainable params only
        params = list(filter(lambda p: p.requires_grad, list(self.decomposer.parameters()) + list(self.shader.parameters())))
        self.optimizer = Adam(params, lr=learning_rate)

        # misc
        self.epochs = epochs
        self.log_folder = log_folder
        self.checkpoints_folder = checkpoints_folder
        Path(self.log_folder).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoints_folder).mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def _make_fg(self, mask3):
        return (mask3 >= 0.25).any(dim=1, keepdim=True)

    def compute_loss(self, batch):
        # batch: mask, I, R, _, N, D, _, _, L  (we only need mask & I for self-supervised)
        mask, I, _, _, _, _, _, _, _ = batch
        I = I.to(self.device)
        mask = mask.to(self.device)

        # forward full composer
        R_hat, D_hat, N_hat, L_hat = self.decomposer(I, mask)
        N_hat = normalize_normals(N_hat)
        S_hat = self.shader(N_hat, L_hat)
        I_hat = R_hat * S_hat

        fg = self._make_fg(mask)
        recon = masked_l1(I_hat, I, fg)

        # (Optional) tiny regularizers
        tv_weight = 1e-4
        tv = (R_hat[:, :, :, 1:] - R_hat[:, :, :, :-1]).abs().mean() + (R_hat[:, :, 1:, :] - R_hat[:, :, :-1, :]).abs().mean()
        loss = recon + tv_weight * tv
        return loss, recon, tv

    def train(self):
        header = ['epoch', 'train_loss', 'train_recon', 'train_tv', 'val_loss', 'val_recon', 'val_tv']
        time = datetime.now().strftime('%Y-%m-%d_%H-%M')
        filename = os.path.join(self.log_folder, f"train_composer_{time}.csv")

        # save decomposer/shader separately under a timestamped folder
        ckpt_root = os.path.join(self.checkpoints_folder, time)
        ckpt_dec = os.path.join(ckpt_root, "decomposer")
        ckpt_sh = os.path.join(ckpt_root, "shader")
        Path(ckpt_dec).mkdir(parents=True, exist_ok=True)
        Path(ckpt_sh).mkdir(parents=True, exist_ok=True)

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for epoch in range(self.epochs):
                # ---- train
                pbar = tqdm.tqdm(total=len(self.train_loader), desc=f"[Composer] Train {epoch}")
                self.decomposer.train()
                self.shader.train()
                total_tr = total_rr = total_tv = 0.0
                for i, batch in enumerate(self.train_loader):
                    self.optimizer.zero_grad(set_to_none=True)
                    loss, recon, tv = self.compute_loss(batch)
                    total_tr += loss.item()
                    total_rr += recon.item()
                    total_tv += tv.item()
                    loss.backward()
                    self.optimizer.step()
                    if i % 100 == 0:
                        pbar.set_description(f"[Comp] Ep{epoch} L:{loss.item():.4f} R:{recon.item():.4f}")
                    pbar.update(1)
                pbar.close()

                # ---- val
                pbar = tqdm.tqdm(total=len(self.validate_loader), desc=f"[Composer] Val   {epoch}")
                self.decomposer.eval()
                self.shader.eval()
                total_vl = total_vr = total_vt = 0.0
                with torch.no_grad():
                    for batch in self.validate_loader:
                        loss, recon, tv = self.compute_loss(batch)
                        total_vl += loss.item()
                        total_vr += recon.item()
                        total_vt += tv.item()
                        pbar.update(1)
                pbar.close()

                # ---- save both modules
                torch.save(self.decomposer.state_dict(), os.path.join(ckpt_dec, f"decomposer_{epoch}.pth"))
                if any(p.requires_grad for p in self.shader.parameters()):
                    torch.save(self.shader.state_dict(), os.path.join(ckpt_sh, f"shader_{epoch}.pth"))

                writer.writerow([
                    epoch,
                    total_tr/len(self.train_loader),
                    total_rr/len(self.train_loader),
                    total_tv/len(self.train_loader),
                    total_vl/len(self.validate_loader),
                    total_vr/len(self.validate_loader),
                    total_vt/len(self.validate_loader),
                ])
                f.flush()

if __name__ == "__main__":
    import tomli
    with open("config.toml", "rb") as cf:
        config = tomli.load(cf)
    ComposerTrainer(config).train()

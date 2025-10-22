"""
Author: Yue (Yelena) Yu, Rongfei (Eric) Jin
Purpose: Training script for Shader model
- Handles training loop for shader network
- Supports multi-light training with comprehensive logging
"""

# train_shader.py
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

from models import NeuralShader, NeuralShaderVariant
from datasets import IntrinsicDataset
from utils import find_lastest_checkpoint #,normalize_normals, masked_l1, compute_shading_gt




class ShaderTrainer:
    def __init__(self, config):
        self.device = config["train"]["device"]
        num_workers = config["train"]["num_workers"]
        print(f"Using device: {self.device}")

        # read config
        scfg = config["train"]["shader"]
        epochs = scfg["epochs"]
        learning_rate = scfg["learning_rate"]
        log_folder = scfg["log_folder"]
        checkpoints_folder = scfg["checkpoints_folder"]
        train_datasets = scfg["train_datasets"]
        validate_datasets = scfg["validate_datasets"]
        light_array = scfg["light_array"]
        load_latest_checkpoint = scfg["load_latest_checkpoint"]
        batch_size = scfg.get("batch_size", 4)
        lights_dim = scfg.get("lights_dim", 4)
        num_lights = scfg.get("num_lights", 1)
        expand_dim = scfg.get("expand_dim", 8)
        use_variant = scfg.get("use_variant", False)

        # datasets
        train_dataset = IntrinsicDataset(train_datasets, light_array)
        validate_dataset = IntrinsicDataset(validate_datasets, light_array)
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Val   dataset size: {len(validate_dataset)}")

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
        self.validate_loader = DataLoader(validate_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)

        # model
        if not use_variant:
            print("Using standard shader")
            self.model = NeuralShader(lights_dim=lights_dim, expand_dim=expand_dim, num_lights=num_lights).to(self.device)
        else:
            print("Using variant shader")
            self.model = NeuralShaderVariant(lights_dim=lights_dim, expand_dim=expand_dim).to(self.device)

        # optim/loss
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.epochs = epochs
        self.checkpoint_number = 0

        # io
        self.log_folder = log_folder
        self.checkpoints_folder = checkpoints_folder
        Path(self.log_folder).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoints_folder).mkdir(parents=True, exist_ok=True)

        if load_latest_checkpoint:
            latest, ckpt_num = find_lastest_checkpoint(self.checkpoints_folder)
            if latest is not None:
                print(f"Loading checkpoint: {latest}")
                print(f"Last Checkpoint number: {ckpt_num}")
                self.model.load_state_dict(torch.load(latest, map_location=self.device))
                self.checkpoint_number = ckpt_num + 1


    def compute_loss(self, batch):
        # batch tuple order from your DecomposerTrainer: 
        # mask, reconstructed(I), reflectance, shading, normals, depth, _, _, lights
        _, _, _, S, N, _, _, _, L = batch
        S = S.to(self.device)
        N = N.to(self.device)
        L = L.to(self.device)

        # predict
        S_hat = self.model(N, L)

        # loss
        return nn.MSELoss()(S_hat, S)

    def train(self):
        header = ['epoch', 'train_loss', 'val_loss']
        time = datetime.now().strftime('%Y-%m-%d_%H-%M')
        filename = os.path.join(self.log_folder, f"train_shader_{time}.csv")
        ckpt_dir = os.path.join(self.checkpoints_folder, time)
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for epoch in range(self.checkpoint_number, self.epochs):
                # train
                pbar = tqdm.tqdm(total=len(self.train_loader), desc=f"[Shader] Train {epoch}")
                self.model.train()
                total_tr = 0.0
                for i, batch in enumerate(self.train_loader):
                    self.optimizer.zero_grad(set_to_none=True)
                    loss = self.compute_loss(batch)
                    total_tr += loss.item()
                    loss.backward()
                    self.optimizer.step()
                    if i % 100 == 0:
                        pbar.set_description(f"[Shader] Ep{epoch} L:{loss.item():.4f}")
                    pbar.update(1)
                pbar.close()

                # val
                pbar = tqdm.tqdm(total=len(self.validate_loader), desc=f"[Shader] Val   {epoch}")
                self.model.eval()
                total_val = 0.0
                with torch.no_grad():
                    for batch in self.validate_loader:
                        loss = self.compute_loss(batch)
                        total_val += loss.item()
                        pbar.update(1)
                pbar.close()

                # save
                torch.save(self.model.state_dict(), os.path.join(ckpt_dir, f"model_{epoch}.pth"))
                writer.writerow([epoch, total_tr/len(self.train_loader), total_val/len(self.validate_loader)])
                f.flush()

if __name__ == "__main__":
    # Expect a small loader that gives you a `config` dict (toml/yaml already loaded)
    import tomli
    with open("config.toml", "rb") as cf:
        config = tomli.load(cf)
    ShaderTrainer(config).train()

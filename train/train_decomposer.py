"""
Author: Yue (Yelena) Yu, Rongfei (Eric) Jin
Purpose: Training script for Decomposer model
- Handles training loop, validation, and checkpointing for the decomposer network
- Supports multi-light training and comprehensive logging
"""

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

from models import Decomposer
from datasets import IntrinsicDataset
from utils.checkpoint import find_lastest_checkpoint

class DecomposerTrainer:
    def __init__(self, config):

        epochs = config["train"]["decomposer"]["epochs"]
        learning_rate = config["train"]["decomposer"]["learning_rate"]
        light_loss_coef = config["train"]["decomposer"]["light_loss_coef"]
        log_folder = config["train"]["decomposer"]["log_folder"]
        checkpoints_folder = config["train"]["decomposer"]["checkpoints_folder"]
        train_datasets = config["train"]["decomposer"]["train_datasets"]
        validate_datasets = config["train"]["decomposer"]["validate_datasets"]
        light_array = config["train"]["decomposer"]["light_array"]
        load_latest_checkpoint = config["train"]["decomposer"]["load_latest_checkpoint"]

        self.device = config["train"]["device"]
        print(f"Using device: {self.device}")


        # Setup datasets and dataloaders
        train_dataset = IntrinsicDataset(train_datasets, light_array)
        validate_dataset = IntrinsicDataset(validate_datasets, light_array)
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Train dataset size: {len(train_dataset)}")

        self.train_loader = DataLoader(train_dataset, batch_size=4, num_workers=2)
        self.validate_loader = DataLoader(validate_dataset, batch_size=4, num_workers=2)

        # Setup model
        lights_dim = config["train"]["decomposer"].get("lights_dim", 4)
        num_lights = config["train"]["decomposer"].get("num_lights", 1)
        self.model = Decomposer(lights_dim=lights_dim, num_lights=num_lights).to(self.device)

        # Setup optimizer and criterion
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.light_loss_coef = light_loss_coef
        self.log_folder = log_folder
        self.checkpoints_folder = checkpoints_folder
        self.epochs = epochs
        self.checkpoint_number = 0

        # Setup log folder and checkpoints folder
        Path(self.log_folder).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoints_folder).mkdir(parents=True, exist_ok=True)

        if load_latest_checkpoint:
            latest_checkpoint, checkpoint_number = find_lastest_checkpoint(self.checkpoints_folder)
            print(f"Loading checkpoint: {latest_checkpoint}")
            print(f"Last Checkpoint number: {checkpoint_number}")
            self.model.load_state_dict(torch.load(latest_checkpoint, map_location=self.device))
            self.checkpoint_number = checkpoint_number + 1

    def compute_loss(self, batch):
        mask, reconstructed, reflectance, _, normals, depth, _, _, lights = batch

        target_reconstructed = reconstructed.to(self.device)
        target_reflectance = reflectance.to(self.device)
        target_mask = mask.to(self.device)
        target_depth = depth.to(self.device)
        target_normals = normals.to(self.device)
        target_lights = lights.to(self.device)

        predicted_reflectance, predicted_depth, predicted_normals, predicted_lights = self.model(target_reconstructed, target_mask)

        # Compute the intrinsic losses
        reflectance_loss = self.criterion(predicted_reflectance, target_reflectance)
        depth_loss = self.criterion(predicted_depth, target_depth)
        normals_loss = self.criterion(predicted_normals, target_normals)
        lights_loss = self.criterion(predicted_lights, target_lights)
        
        # The total loss is the sum of the losses
        return reflectance_loss + depth_loss + normals_loss + lights_loss * self.light_loss_coef

    def train(self):
        header = ['epoch', 'train_loss', 'val_loss']

        time = datetime.now().strftime('%Y-%m-%d_%H-%M')

        filename = f"train_{time}.csv"
        filename = os.path.join(self.log_folder, filename)

        # create a folder with the name of the time
        checkpoint_folder = os.path.join(self.checkpoints_folder, time)
        Path(checkpoint_folder).mkdir(parents=True, exist_ok=True)

        with open(filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for epoch in range(self.checkpoint_number, self.epochs):
                pbar = tqdm.tqdm(total=len(self.train_loader), desc=f"Training epoch {epoch}")

                self.model.train()
                total_training_loss = 0
                for i, batch in enumerate(self.train_loader):
                    self.optimizer.zero_grad()
                    loss = self.compute_loss(batch)
                    total_training_loss += loss.item()
                    loss.backward()
                    self.optimizer.step()

                    if i % 100 == 0:
                        pbar.set_description(f"Epoch {epoch} Loss: {loss.item():.4f}")
                    pbar.update(1)

                pbar.close()

                pbar = tqdm.tqdm(total=len(self.validate_loader), desc=f"Validating epoch {epoch}")

                total_validation_loss = 0

                self.model.eval()
                for batch in self.validate_loader:
                    with torch.no_grad():
                        loss = self.compute_loss(batch)
                        total_validation_loss += loss.item()
                        pbar.set_description(f"Epoch {epoch} Loss: {loss.item()}")
                        pbar.update(1)
                
                pbar.close()
                torch.save(self.model.state_dict(), os.path.join(checkpoint_folder, f"model_{epoch}.pth"))
                writer.writerow([epoch, total_training_loss/len(self.train_loader), total_validation_loss/len(self.validate_loader)])

if __name__ == "__main__":
    # Expect a small loader that gives you a `config` dict (toml/yaml already loaded)
    import tomli
    with open("config.toml", "rb") as cf:
        config = tomli.load(cf)
    DecomposerTrainer(config).train()
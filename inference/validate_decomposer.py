from pathlib import Path
import os
import csv

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import tqdm

from models import Decomposer
from datasets import IntrinsicDataset

class DecomposerValidator:
    def __init__(self, config, date_string):
        self.checkpoints_folder = Path(config["train"]["decomposer"]["checkpoints_folder"]) / date_string

        self.device = config["train"]["device"]
        self.log_folder = config["train"]["decomposer"]["log_folder"]
        self.validate_datasets = config["train"]["decomposer"]["validate_datasets"]
        self.light_array = config["train"]["decomposer"]["light_array"]
        
        self.validate_dataloader = DataLoader(IntrinsicDataset(self.validate_datasets, self.light_array), batch_size=1, shuffle=False)
        self.light_loss_coef = config["train"]["decomposer"].get("light_loss_coef", 1)

        lights_dim = config["train"]["decomposer"].get("lights_dim", 4)
        num_lights = config["train"]["decomposer"].get("num_lights", 1)
        self.model = Decomposer(lights_dim=lights_dim, num_lights=num_lights).to(self.device)

        self.criterion = nn.MSELoss()

        self.writer = csv.writer(open(os.path.join(self.log_folder, f"validate_{self.checkpoints_folder.stem}.csv"), "w"))
        self.writer.writerow(["epoch", "validate_loss", "reflectance_loss", "depth_loss", "normals_loss", "lights_loss"])

    def validate(self):
        checkpoints = list(Path(self.checkpoints_folder).glob("*.pth"))
        checkpoints.sort(key=lambda x: int(x.stem.split("_")[-1]))
        for checkpoint_path in checkpoints:
            epoch = checkpoint_path.stem.split("_")[-1]
            average_loss, average_reflectance_loss, average_depth_loss, average_normals_loss, average_lights_loss = self._validate(checkpoint_path)
            self.writer.writerow([epoch, average_loss, average_reflectance_loss, average_depth_loss, average_normals_loss, average_lights_loss])

    def _validate(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            total_reflectance_loss = 0
            total_depth_loss = 0
            total_normals_loss = 0
            total_lights_loss = 0
            for batch in tqdm.tqdm(self.validate_dataloader, total=len(self.validate_dataloader)):
                mask, reconstructed, reflectance, _, normals, depth, _, _, lights = batch

                target_reconstructed = reconstructed.to(self.device)
                target_reflectance = reflectance.to(self.device)
                target_mask = mask.to(self.device)
                target_depth = depth.to(self.device)
                target_normals = normals.to(self.device)
                target_lights = lights.to(self.device)

                predicted_reflectance, predicted_depth, predicted_normals, predicted_lights = self.model(target_reconstructed, target_mask)

                reflectance_loss = self.criterion(predicted_reflectance, target_reflectance)
                depth_loss = self.criterion(predicted_depth, target_depth)
                normals_loss = self.criterion(predicted_normals, target_normals)
                lights_loss = self.criterion(predicted_lights, target_lights)

                loss = reflectance_loss + depth_loss + normals_loss + lights_loss * self.light_loss_coef

                total_loss += loss.item()
                total_reflectance_loss += reflectance_loss.item()
                total_depth_loss += depth_loss.item()
                total_normals_loss += normals_loss.item()
                total_lights_loss += lights_loss.item() * self.light_loss_coef
        return total_loss / len(self.validate_dataloader), total_reflectance_loss / len(self.validate_dataloader), total_depth_loss / len(self.validate_dataloader), total_normals_loss / len(self.validate_dataloader), total_lights_loss / len(self.validate_dataloader)
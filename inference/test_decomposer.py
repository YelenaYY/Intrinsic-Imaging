from pathlib import Path

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from models import Decomposer
from datasets import IntrinsicDataset
from utils.checkpoint import find_lastest_checkpoint



class DecomposerTester:
    def __init__(self, config):
        self.checkpoints_folder = config["train"]["decomposer"]["checkpoints_folder"]
        self.device = config["train"]["device"]
        self.output_folder = config["test"]["decomposer"]["output_folder"]
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)

        self.test_datasets = config["test"]["decomposer"]["test_datasets"]
        self.light_array = config["train"]["decomposer"]["light_array"]
        
        self.test_dataloader = DataLoader(IntrinsicDataset(self.test_datasets, self.light_array), batch_size=1, shuffle=False)

        lights_dim = config["train"]["decomposer"].get("lights_dim", 4)
        num_lights = config["train"]["decomposer"].get("num_lights", 1)
        self.model = Decomposer(lights_dim=lights_dim, num_lights=num_lights).to(self.device)

        latest_checkpoint, checkpoint_number = find_lastest_checkpoint(self.checkpoints_folder)
        print(f"Loading checkpoint: {latest_checkpoint}")
        print(f"Checkpoint number: {checkpoint_number}")
        self.model.load_state_dict(torch.load(latest_checkpoint))


    def test(self):
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.test_dataloader):
                mask, reconstructed, reflectance, _, normals, depth, _, _, lights = batch

                target_reconstructed = reconstructed.to(self.device)
                target_reflectance = reflectance.to(self.device)
                target_mask = mask.to(self.device)
                target_depth = depth.to(self.device)
                target_normals = normals.to(self.device)
                target_lights = lights.to(self.device)

                predicted_reflectance, predicted_depth, predicted_normals, predicted_lights = self.model(target_reconstructed, target_mask)

                # move to cpu
                predicted_reflectance = predicted_reflectance.cpu().detach().squeeze(0)
                predicted_depth = predicted_depth.cpu().detach().squeeze(0)
                predicted_normals = predicted_normals.cpu().detach().squeeze(0)
                predicted_lights = predicted_lights.cpu().detach().squeeze(0)
                target_reflectance = target_reflectance.cpu().detach().squeeze(0)
                target_depth = target_depth.cpu().detach().squeeze(0)
                target_normals = target_normals.cpu().detach().squeeze(0)
                target_lights = target_lights.cpu().detach().squeeze(0)

                self.save_images(i, predicted_reflectance, predicted_depth, predicted_normals, predicted_lights, target_reflectance, target_depth, target_normals, target_lights)

    
    def save_images(self, i, predicted_reflectance, predicted_depth, predicted_normals, predicted_lights, target_reflectance, target_depth, target_normals, target_lights):
        fig, axs = plt.subplots(3, 2, figsize=(10, 10))
        axs[0,0].imshow(target_reflectance.permute(1, 2, 0).numpy())
        axs[0,0].set_title("Target Reflectance")
        axs[1,0].imshow(target_depth.permute(1, 2, 0).numpy())
        axs[1,0].set_title("Target Depth")
        axs[2,0].imshow(target_normals.permute(1, 2, 0).numpy())
        axs[2,0].set_title("Target Normals")

        axs[0,1].imshow(predicted_reflectance.permute(1, 2, 0).numpy())
        axs[0,1].set_title("Predicted Reflectance")
        axs[1,1].imshow(predicted_depth.permute(1, 2, 0).numpy())
        axs[1,1].set_title("Predicted Depth")
        axs[2,1].imshow(predicted_normals.permute(1, 2, 0).numpy())
        axs[2,1].set_title("Predicted Normals")

        fig.suptitle(f"Image {i}, target lights: {target_lights}, predicted lights: {predicted_lights}")
        fig.tight_layout()

        fig.savefig(f"{self.output_folder}/test_{i}.png")
        plt.close(fig)


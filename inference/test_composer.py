from pathlib import Path

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from models import Decomposer, NeuralShader, NeuralShaderVariant, Composer
from datasets import IntrinsicDataset
from utils.checkpoint import find_lastest_checkpoint


class ComposerTester:
    def __init__(self, config):
        self.checkpoints_folder = config["train"]["composer"]["checkpoints_folder"]
        self.device = config["train"]["device"]
        self.output_folder = config["test"]["composer"]["output_folder"]
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)

        # remove files in output folder
        for file in Path(self.output_folder).glob("*.png"):
            file.unlink()

        self.test_datasets = config["test"]["composer"]["test_datasets"]
        self.light_array = config["train"]["composer"]["light_array"]
        self.use_shader_variant = config["train"]["composer"]["use_shader_variant"]
        lights_dim = config["train"]["composer"].get("lights_dim", 4)
        num_lights = config["train"]["composer"].get("num_lights", 1)
        self.max_num_images_per_dataset = config["test"]["composer"]["max_num_images_per_dataset"]

        self.test_dataloader = DataLoader(IntrinsicDataset(self.test_datasets, self.light_array, max_num_images_per_dataset=self.max_num_images_per_dataset), batch_size=1, shuffle=False)

        if self.use_shader_variant and num_lights != 1:
            print("shader_variant only supports a single light. Falling back to standard shader for multiple lights.")
            self.use_shader_variant = False

        if self.use_shader_variant:
            shader = NeuralShaderVariant(lights_dim=4).to(self.device)
        else:
            shader = NeuralShader(lights_dim=lights_dim, num_lights=num_lights).to(self.device)
        decomposer = Decomposer(lights_dim=lights_dim, num_lights=num_lights).to(self.device)
        self.model = Composer(shader, decomposer).to(self.device)

        if "learned_composer_checkpoint" in config["test"]["composer"]:
            print(f"Loading learned composer checkpoint: {config['test']['composer']['learned_composer_checkpoint']}")
            self.model.load_state_dict(torch.load(config["test"]["composer"]["learned_composer_checkpoint"], map_location=self.device))
        else:
            latest_checkpoint, checkpoint_number = find_lastest_checkpoint(self.checkpoints_folder)
            print(f"Loading checkpoint: {latest_checkpoint}")
            print(f"Checkpoint number: {checkpoint_number}")
            try:
                self.model.load_state_dict(torch.load(latest_checkpoint, map_location=self.device))
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Check if the checkpoint shader is standard or variant")
                raise e



    def test(self):
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.test_dataloader):
                mask, reconstructed, reflectance, shading, normals, depth, _, _, lights = batch

                target_reconstructed = reconstructed.to(self.device)
                target_reflectance = reflectance.to(self.device)
                target_shading = shading.to(self.device)
                target_mask = mask.to(self.device)
                target_depth = depth.to(self.device)
                target_normals = normals.to(self.device)
                target_lights = lights.to(self.device)

                prediction = self.model(target_reconstructed, target_mask)

                # move to cpu
                predicted_reconstructed = prediction["reconstructed"].cpu().detach().squeeze(0)
                predicted_reflectance = prediction["reflectance"].cpu().detach().squeeze(0)
                predicted_normals = prediction["normals"].cpu().detach().squeeze(0)
                predicted_depth = prediction["depth"].cpu().detach().squeeze(0)
                predicted_lights = prediction["lights"].cpu().detach().squeeze(0)
                predicted_shading = prediction["shading"].cpu().detach().squeeze(0)

                target_reconstructed = target_reconstructed.cpu().detach().squeeze(0)
                target_reflectance = target_reflectance.cpu().detach().squeeze(0)
                target_depth = target_depth.cpu().detach().squeeze(0)
                target_normals = target_normals.cpu().detach().squeeze(0)
                target_lights = target_lights.cpu().detach().squeeze(0)
                target_shading = target_shading.cpu().detach().squeeze(0)

                self.save_images(i, predicted_reconstructed, predicted_reflectance, predicted_depth, predicted_normals, predicted_lights, predicted_shading, target_reconstructed, target_reflectance, target_depth, target_normals, target_lights, target_shading)

    
    def save_images(self, i, predicted_reconstructed, predicted_reflectance, predicted_depth, predicted_normals, predicted_lights, predicted_shading, target_reconstructed, target_reflectance, target_depth, target_normals, target_lights, target_shading):
        fig, axs = plt.subplots(5, 2, figsize=(10, 10))
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

        axs[3,0].imshow(target_reconstructed.permute(1, 2, 0).numpy())
        axs[3,0].set_title("Target Reconstructed")
        axs[3,1].imshow(predicted_reconstructed.permute(1, 2, 0).numpy())
        axs[3,1].set_title("Predicted Reconstructed")

        axs[4,0].imshow(target_shading.permute(1, 2, 0).numpy())
        axs[4,0].set_title("Target Shading")
        axs[4,1].imshow(predicted_shading.permute(1, 2, 0).numpy())
        axs[4,1].set_title("Predicted Shading")

        fig.suptitle(f"Image {i}, target lights: {target_lights}, predicted lights: {predicted_lights}")
        fig.tight_layout()

        fig.savefig(f"{self.output_folder}/test_{i}.png")
        plt.close(fig)


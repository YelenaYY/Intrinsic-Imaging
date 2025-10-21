from pathlib import Path
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from models import NeuralShader, NeuralShaderVariant
from datasets import IntrinsicDataset
from utils import find_lastest_checkpoint, normalize_normals, masked_l1, compute_shading_gt, mask_image



class ShaderTester:
    def __init__(self, config):
        self.checkpoints_folder = config["train"]["shader"]["checkpoints_folder"]
        self.device = config["train"]["device"]
        self.output_folder = config["test"]["shader"]["output_folder"]
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)

        self.test_datasets = config["test"]["shader"]["test_datasets"]
        self.light_array = config["train"]["shader"]["light_array"]
        self.use_variant = config["train"]["shader"].get("use_variant", False)
        self.learned_shader_checkpoint = config["test"]["shader"].get("learned_shader_checkpoint", None)

        self.test_dataloader = DataLoader(IntrinsicDataset(self.test_datasets, self.light_array, max_num_images_per_dataset=50), batch_size=1, shuffle=False)

        if self.use_variant:
            print("Using variant shader")
            self.model = NeuralShaderVariant(lights_dim=4).to(self.device)
        else:
            print("Using standard shader")
            self.model = NeuralShader(lights_dim=4).to(self.device)

        if self.learned_shader_checkpoint is not None:
            print(f"Loading learned shader checkpoint: {self.learned_shader_checkpoint}")
            self.model.load_state_dict(torch.load(self.learned_shader_checkpoint))
        else:   
            latest_checkpoint, checkpoint_number = find_lastest_checkpoint(self.checkpoints_folder)
            print(f"Loading checkpoint: {latest_checkpoint}")
            print(f"Checkpoint number: {checkpoint_number}")
            self.model.load_state_dict(torch.load(latest_checkpoint))


    def test(self):
        self.model.eval()
        with torch.no_grad():
            for i, batch in tqdm.tqdm(enumerate(self.test_dataloader), total=len(self.test_dataloader)):
                _, _, _, S, N, _, _, _, L = batch

                # mask = mask.to(self.device)
                # I = I.to(self.device)
                # R = R.to(self.device)
                S = S.to(self.device)
                N = N.to(self.device)
                L = L.to(self.device)


                # targets
                # S_gt, fg = compute_shading_gt(I, R, mask)
                # N = normalize_normals(N)
                # predict
                S_hat = self.model(N, L)

                loss = nn.MSELoss()(S_hat, S)

                # S_hat_masked = mask_image(S_hat, fg)

                # S_gt = S_gt.cpu().detach().squeeze(0)
                # S_hat = S_hat.cpu().detach().squeeze(0)
                # S_hat_masked = S_hat_masked.cpu().detach().squeeze(0)
                S_hat = S_hat.cpu().detach().squeeze(0)
                S = S.cpu().detach().squeeze(0)
                loss = loss.cpu().detach()

                self.save_images(i, S, S_hat, loss)

    
    def save_images(self, i, target_shading, predicted_shading, loss):
        fig, axs = plt.subplots(1, 2, figsize=(10, 10))
        axs[0].imshow(target_shading.permute(1, 2, 0).numpy(), cmap="gray")
        axs[0].set_title("Target Shading")
        axs[1].imshow(predicted_shading.permute(1, 2, 0).numpy(), cmap="gray")
        axs[1].set_title("Predicted Shading")
        fig.suptitle(f"Image {i}, loss: {loss.item()}")
        fig.tight_layout()
        fig.savefig(f"{self.output_folder}/test_{i}.png")
        plt.close(fig)

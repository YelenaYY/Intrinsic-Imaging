# train_composer.py
import os
import sys
from pathlib import Path
from datetime import datetime
import csv
from copy import deepcopy

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm

from models import Composer, Decomposer, NeuralShader, NeuralShaderVariant
from datasets import ComposerDataset, IntrinsicDataset
from utils.checkpoint import find_lastest_checkpoint

def normalize_normals(n):
    mag = n.pow(2).sum(1, keepdim=True).sqrt().clamp_min(1e-6)
    return n / mag

def masked_l1(pred, target, mask_bool):
    m = mask_bool.expand_as(pred)
    num = m.sum().clamp_min(1)
    return (pred[m] - target[m]).abs().sum() / num

# Set what parameters to update based on the learning schedule (e.g. update normals decoder from epoch 0 to 30)
def extract_learning_schedule(schedule, epoch, lr, decomposer, shader):
    parameters = []
    for item in schedule:
        image_type, start_epoch, end_epoch = item
        if epoch < start_epoch or epoch > end_epoch:
            continue

        if image_type == 'reflectance':
            parameters.append( {'params': decomposer.decoder_reflectance.parameters(), 'lr': lr} )
        if image_type == 'normals':
            parameters.append( {'params': decomposer.decoder_normals.parameters(), 'lr': lr} )
        if image_type == 'lights':
            parameters.append( {'params': decomposer.lights_encoder.parameters(), 'lr': lr} )
        if image_type == 'shader':
            parameters.append( {'params': shader.parameters(), 'lr': lr} )

    return parameters


class ComposerTrainer:
    def __init__(self, config, transfer_type):
        self.device = config["train"]["device"]
        print(f"Using device: {self.device}")

        composer_config = deepcopy(config["train"]["composer"])
        if transfer_type == "shape":
            print("Using shape specific config")
            composer_config.update(config["train"]["composer"]["shape"])
        elif transfer_type == "category":
            composer_config.update(config["train"]["composer"]["category"])
        else:
            raise ValueError(f"Invalid transfer type: {transfer_type}")

        # config
        epochs = composer_config["epochs"]
        print(f"Using {epochs} epochs")

        learning_rate = composer_config["learning_rate"]
        print(f"Using {learning_rate} learning rate")

        checkpoints_folder = composer_config["checkpoints_folder"]
        log_folder = composer_config["log_folder"]

        labeled_train_datasets = composer_config["labeled_train_datasets"]
        unlabeled_train_datasets = composer_config["unlabeled_train_datasets"]
        unlabeled_validate_datasets = composer_config["unlabeled_validate_datasets"]

        max_num_images_per_dataset = composer_config.get("max_num_images_per_dataset", None)

        light_array = composer_config["light_array"]

        batch_size = composer_config.get("batch_size", 4)
        lights_dim = composer_config.get("lights_dim", 4)
        expand_dim = composer_config.get("expand_dim", 8)
        num_lights = composer_config.get("num_lights", 1)

        use_shader_variant = composer_config["use_shader_variant"]
        learned_shader_checkpoint = composer_config["learned_shader_checkpoint"]
        learned_decomposer_checkpoint = composer_config["learned_decomposer_checkpoint"]

        if not Path(learned_shader_checkpoint).exists() or not Path(learned_decomposer_checkpoint).exists():
            raise ValueError("Learned shader or decomposer checkpoint not found")
        
        if use_shader_variant and num_lights != 1:
            print("shader_variant only supports a single light. Falling back to standard shader for multiple lights.")
            use_shader_variant = False

        if use_shader_variant:
            shader = NeuralShaderVariant(lights_dim=lights_dim, expand_dim=expand_dim).to(self.device)
        else:
            shader = NeuralShader(lights_dim=lights_dim, expand_dim=expand_dim, num_lights=num_lights).to(self.device)

        decomposer = Decomposer(lights_dim=lights_dim, num_lights=num_lights).to(self.device)

        self.model = Composer(shader, decomposer).to(self.device)

        if composer_config["load_latest_checkpoint"]:
            last_checkpoint_path, last_checkpoint_number = find_lastest_checkpoint(checkpoints_folder)
            print(f"Loading composer checkpoint: {last_checkpoint_path}")
            print(f"Last Checkpoint number: {last_checkpoint_number}")
            self.model.load_state_dict(torch.load(last_checkpoint_path, map_location=self.device))
            self.checkpoint_number = last_checkpoint_number + 1
        else:
            print("Training from scratch (still use provided shader and decomposer checkpoints)")
            shader.load_state_dict(torch.load(learned_shader_checkpoint, map_location=self.device))
            decomposer.load_state_dict(torch.load(learned_decomposer_checkpoint, map_location=self.device))
            self.checkpoint_number = 0
        
        # datasets
        self.composer_train_dataset = ComposerDataset(labeled_train_datasets, unlabeled_train_datasets, light_array, max_num_images_per_dataset=max_num_images_per_dataset)
        self.validate_dataset = IntrinsicDataset(unlabeled_validate_datasets, light_array)

        # dataloaders
        num_workers = config["train"]["num_workers"]
        print(f"Using {num_workers} workers for data loading")
        self.composer_train_loader = DataLoader(self.composer_train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
        self.composer_validate_loader = DataLoader(self.validate_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)

        # learning schedules
        self.schedule = composer_config["schedule"]
        self.learning_rate = learning_rate

        self.criterion = nn.MSELoss()
        self.unlabel_loss_coef = composer_config.get("unlabel_loss_coef", 1.0)
        self.labeled_loss_coef = composer_config.get("labeled_loss_coef", 1.0)
        self.light_loss_coef = composer_config.get("light_loss_coef", 1.0)

        print(f"Using unlabel loss coef: {self.unlabel_loss_coef}")
        print(f"Using labeled loss coef: {self.labeled_loss_coef}")
        print(f"Using light loss coef: {self.light_loss_coef}")

        # misc
        self.epochs = epochs
        self.log_folder = log_folder
        self.checkpoints_folder = checkpoints_folder
        Path(self.log_folder).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoints_folder).mkdir(parents=True, exist_ok=True)


    # @torch.no_grad()
    # def _make_fg(self, mask3):
    #     return (mask3 >= 0.25).any(dim=1, keepdim=True)

    # def compute_loss(self, batch):
    #     # batch: mask, I, R, _, N, D, _, _, L  (we only need mask & I for self-supervised)
    #     mask, I, _, _, _, _, _, _, _ = batch
    #     I = I.to(self.device)
    #     mask = mask.to(self.device)

    #     # forward full composer
    #     R_hat, D_hat, N_hat, L_hat = self.decomposer(I, mask)
    #     N_hat = normalize_normals(N_hat)
    #     S_hat = self.shader(N_hat, L_hat)
    #     I_hat = R_hat * S_hat

    #     fg = self._make_fg(mask)
    #     recon = masked_l1(I_hat, I, fg)

    #     # (Optional) tiny regularizers
    #     tv_weight = 1e-4
    #     tv = (R_hat[:, :, :, 1:] - R_hat[:, :, :, :-1]).abs().mean() + (R_hat[:, :, 1:, :] - R_hat[:, :, :-1, :]).abs().mean()
    #     loss = recon + tv_weight * tv
    #     return loss, recon, tv

    def compute_loss(self, batch):
        labeled, unlabeled = batch

        unlabeled_mask, unlabeled_reconstructed, _, _, _, _, _, _, _ = unlabeled
        labeled_mask, labeled_reconstructed, labeled_reflectance, labeled_shading, labeled_normals, labeled_depth, _, _, labeled_lights = labeled

        unlabeled_mask = unlabeled_mask.to(self.device)
        unlabeled_reconstructed = unlabeled_reconstructed.to(self.device)

        labeled_mask = labeled_mask.to(self.device)
        labeled_reconstructed = labeled_reconstructed.to(self.device)
        labeled_reflectance = labeled_reflectance.to(self.device)
        labeled_shading = labeled_shading.to(self.device)
        labeled_normals = labeled_normals.to(self.device)
        labeled_depth = labeled_depth.to(self.device)
        # labeled_specular = labeled_specular.to(self.device)
        # labeled_composite = labeled_composite.to(self.device)
        labeled_lights = labeled_lights.to(self.device)

        # unlabel prediction
        unlabeled_prediction = self.model(unlabeled_reconstructed, unlabeled_mask)
        # labeled prediction
        labeled_prediction = self.model(labeled_reconstructed, labeled_mask)

        # unlabel loss
        unlabel_loss = self.criterion(unlabeled_prediction["reconstructed"], unlabeled_reconstructed)

        # labeled loss
        reflectance_loss = self.criterion(labeled_prediction["reflectance"], labeled_reflectance)
        shading_loss = self.criterion(labeled_prediction["shading"], labeled_shading)
        normals_loss = self.criterion(labeled_prediction["normals"], labeled_normals)
        lights_loss = self.criterion(labeled_prediction["lights"], labeled_lights)

        labeled_loss = reflectance_loss + shading_loss + normals_loss + lights_loss * self.light_loss_coef

        loss = self.unlabel_loss_coef * unlabel_loss + self.labeled_loss_coef * labeled_loss

        return loss
    
    def compute_val_loss(self, batch):
        mask, reconstructed, reflectance, shading, normals, depth, _, _, lights = batch

        mask = mask.to(self.device)
        reconstructed = reconstructed.to(self.device)
        reflectance = reflectance.to(self.device)
        shading = shading.to(self.device)
        normals = normals.to(self.device)
        depth = depth.to(self.device)
        lights = lights.to(self.device)

        predicted_reflectance, predicted_depth, predicted_normals, predicted_lights = self.model.decomposer(reconstructed, mask)

        reflectance_loss = self.criterion(predicted_reflectance, reflectance)
        depth_loss = self.criterion(predicted_depth, depth)
        normals_loss = self.criterion(predicted_normals, normals)
        lights_loss = self.criterion(predicted_lights, lights)

        val_loss = reflectance_loss + depth_loss + normals_loss + lights_loss * self.light_loss_coef
        return val_loss, reflectance_loss, depth_loss, normals_loss, lights_loss


    def train(self):
        header = ['epoch', 'train_loss', 'val_loss', 'reflectance_loss', 'depth_loss', 'normals_loss', 'lights_loss']
        time = datetime.now().strftime('%Y-%m-%d_%H-%M')
        filename = os.path.join(self.log_folder, f"train_composer_{time}.csv")

        ckpt_root = os.path.join(self.checkpoints_folder, time)
        Path(ckpt_root).mkdir(parents=True, exist_ok=True)

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for epoch in range(self.checkpoint_number, self.epochs):
                # ---- train
                parameters = extract_learning_schedule(self.schedule, epoch, self.learning_rate, self.model.decomposer, self.model.shader)
                print(f"Updating {(len(parameters))} decoders based on learning schedule")

                pbar = tqdm.tqdm(total=len(self.composer_train_loader), desc=f"[Composer] Train {epoch}")
                optimizer = Adam(parameters)

                # total_tr = total_rr = total_tv = 0.0
                total_training_loss = 0.0
                for i, batch in enumerate(self.composer_train_loader):
                    optimizer.zero_grad()

                    # loss, recon, tv = self.compute_loss(batch)
                    # total_tr += loss.item()
                    # total_rr += recon.item()
                    # total_tv += tv.item()

                    loss = self.compute_loss(batch)
                    total_training_loss += loss.item()

                    loss.backward()
                    optimizer.step()
                    if i % 100 == 0:
                        pbar.set_description(f"[Comp] Ep{epoch} L:{loss.item():.4f}")
                    pbar.update(1)
                pbar.close()

                torch.save(self.model.state_dict(), os.path.join(ckpt_root, f"model_{epoch}.pth"))

                # ---- val
                pbar = tqdm.tqdm(total=len(self.composer_validate_loader), desc=f"[Composer] Val   {epoch}")
                self.model.decomposer.eval()
                total_validation_loss = 0.0
                total_reflectance_loss = 0.0
                total_depth_loss = 0.0
                total_normals_loss = 0.0
                total_lights_loss = 0.0
                with torch.no_grad():
                    for batch in self.composer_validate_loader:
                        # loss, recon, tv = self.compute_loss(batch)
                        # total_vl += loss.item()
                        # total_vr += recon.item()
                        # total_vt += tv.item()
                        loss, reflectance_loss, depth_loss, normals_loss, lights_loss = self.compute_val_loss(batch)
                        total_validation_loss += loss.item()
                        total_reflectance_loss += reflectance_loss.item()
                        total_depth_loss += depth_loss.item()
                        total_normals_loss += normals_loss.item()
                        total_lights_loss += lights_loss.item()
                        pbar.update(1)
                pbar.close()

                # ---- save both modules
                writer.writerow([
                    epoch,
                    total_training_loss/len(self.composer_train_loader),
                    total_validation_loss/len(self.composer_validate_loader),
                    total_reflectance_loss/len(self.composer_validate_loader),
                    total_depth_loss/len(self.composer_validate_loader),
                    total_normals_loss/len(self.composer_validate_loader),
                    total_lights_loss/len(self.composer_validate_loader),
                ])
                f.flush()

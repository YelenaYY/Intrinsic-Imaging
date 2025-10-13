from torch.optim import Adam
import torch
import torch.nn as nn
import tqdm
import csv
from datetime import datetime
import os
from pathlib import Path


class DecomposeTrainer:
    def __init__(self, model, epochs, learning_rate, light_loss_coef, log_folder, checkpoints_folder):
        self.model = model
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()
        self.light_loss_coef = light_loss_coef
        self.log_folder = log_folder
        self.checkpoints_folder = checkpoints_folder
        self.epochs = epochs

        Path(self.log_folder).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoints_folder).mkdir(parents=True, exist_ok=True)

    def loss_fn(self, predicted_reflectance, predicted_depth, predicted_normals, predicted_lights, target_reflectance, target_depth, target_normals, target_lights):

        reflectance_loss = self.criterion(predicted_reflectance, target_reflectance)
        depth_loss = self.criterion(predicted_depth, target_depth)
        normals_loss = self.criterion(predicted_normals, target_normals)
        lights_loss = self.criterion(predicted_lights, target_lights)
        
        # The total loss is the sum of the losses
        return reflectance_loss + depth_loss + normals_loss + lights_loss * self.light_loss_coef

    def train(self, train_loader):
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
            for epoch in range(self.epochs):
                pbar = tqdm.tqdm(total=len(train_loader), desc=f"Training epoch {epoch}")

                self.model.train()
                for i, batch in enumerate(train_loader):
                    self.optimizer.zero_grad()
                    mask, reconstructed, reflectance, _, normals, depth, _, _, lights = batch
                    img = reconstructed.to(self.device)
                    reflectance = reflectance.to(self.device)
                    mask = mask.to(self.device)
                    depth = depth.to(self.device)
                    normals = normals.to(self.device)
                    lights = lights.to(self.device)
                    predicted_reflectance, predicted_depth, predicted_normals, predicted_lights = self.model(img, mask)
                    loss = self.loss_fn(
                        predicted_reflectance, predicted_depth, predicted_normals, predicted_lights,
                        reflectance, depth, normals, lights
                    )

                    loss.backward()
                    self.optimizer.step()

                    if i % 100 == 0:
                        writer.writerow([epoch, loss.item()])
                    pbar.update(1)
                    pbar.set_description(f"Training loss: {loss.item()}")

                torch.save(self.model.state_dict(), os.path.join(checkpoint_folder, f"model_{epoch}.pth"))

            pbar.close()

    
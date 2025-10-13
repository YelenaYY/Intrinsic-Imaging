import argparse
import tomli

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="decomposer")
parser.add_argument("--config", type=str, default="config.toml")
args = parser.parse_args()




def decompose_trainer(config):
    from train import DecomposeTrainer
    epochs = config["train"]["decomposer"]["epochs"]
    learning_rate = config["train"]["decomposer"]["learning_rate"]
    light_loss_coef = config["train"]["decomposer"]["light_loss_coef"]
    log_folder = config["train"]["decomposer"]["log_folder"]
    checkpoints_folder = config["train"]["decomposer"]["checkpoints_folder"]
    train_datasets = config["train"]["decomposer"]["train_datasets"]
    light_array = config["train"]["decomposer"]["light_array"]
    device = config["train"]["device"]

    from models import Decomposer
    from datasets import IntrinsicDataset
    from torch.utils.data import DataLoader
    model = Decomposer(
        lights_dim=4
    )
    model.to(device)
    dataset = IntrinsicDataset(
        dataset_paths=train_datasets,
        light_path=light_array
    )
    dataloader = DataLoader(dataset, batch_size=4, num_workers=1)
    trainer = DecomposeTrainer(model, epochs, learning_rate, light_loss_coef, log_folder, checkpoints_folder)
    trainer.train(dataloader)

if __name__ == "__main__":
    with open(args.config, "rb") as f:
        config = tomli.load(f)
        if args.model == "decomposer":
            decompose_trainer(config)


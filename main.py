import argparse
import tomli

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="decomposer")
parser.add_argument("--config", type=str, default="config.toml")
parser.add_argument("--test", action="store_true")
args = parser.parse_args()


def decompose_trainer(config):
    from train import DecomposerTrainer
    trainer = DecomposerTrainer(config)
    trainer.train()

def decompose_tester(config):
    from inference import DecomposerTester
    tester = DecomposerTester(config)
    tester.test()

def composer_trainer(config):
    from train import ComposerTrainer
    trainer = ComposerTrainer(config)
    trainer.train()

def composer_tester(config):
    from inference import ComposerTester
    tester = ComposerTester(config)
    tester.test()

def shader_trainer(config):
    from train import ShaderTrainer
    trainer = ShaderTrainer(config)
    trainer.train()

def shader_tester(config):
    from inference import ShaderTester
    tester = ShaderTester(config)
    tester.test()

if __name__ == "__main__":
    with open(args.config, "rb") as f:
        config = tomli.load(f)

        if config["train"]["device"] == "cuda":
            import torch
            # check if cuda is available
            if not torch.cuda.is_available():
                print("CUDA is not available")
                exit(1)

        if args.model == "decomposer":
            if args.test:
                print("Testing...")
                decompose_tester(config)
            else:
                print("Training...")
                decompose_trainer(config)
        elif args.model == "composer":
            if args.test:
                print("Testing...")
                composer_tester(config)
            else:
                print("Training...")
                composer_trainer(config)
        elif args.model == "shader":
            if args.test:
                print("Testing...")
                shader_tester(config)
            else:
                print("Training...")
                shader_trainer(config)
        else:
            print("Invalid model")
            exit(1)

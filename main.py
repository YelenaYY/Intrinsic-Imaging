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

if __name__ == "__main__":
    with open(args.config, "rb") as f:
        config = tomli.load(f)
        if args.model == "decomposer":
            if args.test:
                print("Testing...")
                decompose_tester(config)
            else:
                print("Training...")
                decompose_trainer(config)


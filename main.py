import argparse
import tomli

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="decomposer")
parser.add_argument("--config", type=str, default="config.toml")
parser.add_argument("--test", action="store_true")
parser.add_argument("--validate", type=str, default=None)
args = parser.parse_args()

def print_equal_signs(n=32):
    print("=" * n)

def decompose_trainer(config):
    from train import DecomposerTrainer
    trainer = DecomposerTrainer(config)
    trainer.train()

def decompose_tester(config):
    from inference import DecomposerTester
    tester = DecomposerTester(config)
    tester.test()

def decompose_validator(config, date_string):
    from inference import DecomposerValidator
    validator = DecomposerValidator(config, date_string)
    validator.validate()

def composer_trainer(config):
    from train import ComposerTrainer

    types_to_train = config["train"]["composer"]["types_to_train"]
    print(f"Training {types_to_train} types")
    for transfer_type in types_to_train:
        print_equal_signs()
        print(f"Now training {transfer_type} type...")
        trainer = ComposerTrainer(config, transfer_type)
        trainer.train()

def composer_tester(config):
    from inference import ComposerTester
    types_to_test = config["test"]["composer"]["types_to_test"]
    print(f"Testing {types_to_test} types")
    for transfer_type in types_to_test:
        print_equal_signs()
        print(f"Now testing {transfer_type} type...")
        tester = ComposerTester(config, transfer_type)
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
            elif args.validate is not None:
                print("Validating...")
                decompose_validator(config, args.validate)
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

# Intrinsic-Imaging
Replication of "Self-Supervised Intrinsic Image Decomposition".


# Environment Setup
```bash
conda env create -f environment.yml
```

To downoad datasets, run the following

```bash
bash download_data.sh motorbike car airplane suzanne teapot bunny
```

Primitives shapes required for composer can be downloaded at the following site:

[RIN](https://huggingface.co/datasets/greasycat/rin)

Please download and extract to `datasets/output`

# configurations
We provided default configuration files (config.toml and config_multi.toml)

You might need to adjust a few settings (directory name, epoch size) for own use

# Training and testing
After downloading all necessary files, one can start training 

Single light source
```bash
# To train and test decomposer 
python main.py --model decomposer # train
python main.py --model decomposer --test # test

# To train and test shader
python main.py --model shader # train
python main.py --model shader --test # test

# To train and test composer
# Two composer type "shape", "category" can be trained
# Adjust the config.toml to select which to train
python main.py --model composer # train
python main.py --model composer --test # test
```

To train 2 light sources data add `--config config_multi.toml` to command line 

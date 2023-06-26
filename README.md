# FoundationChestCT
Foundation Model for Chest CT

### Model weights 

https://drive.google.com/drive/folders/1Ks_FZ3L7v2OPnsA1fbKyC0pSH9-_0gMn?usp=drive_link

## Training code
Please clone and install VAE library from https://github.com/AntixK/PyTorch-VAE.git
Place vqvae_bn.py file in VAE/models/
Replace VAE/models/__init__.py with included __inint__.py

The updated model file includes modifications made to the original vaector quantized VAE model in terms of batch normalization

train.py contains code for self-supervised training through masked image pregion prediction.

Dataloader is provided, please plug in your own data file (csv format) with the following columns
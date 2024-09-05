# Propaganda-Evaluation

## Setup
### Environment
### Packages
The whole setup was tested using: 
- Python 3.12
- CUDA 12.1 
- Torch 2.3.1
- huggingface/pytorch-pretrained-BERT **0.4**

### Used Hardware
- RTX 3080 - 10GB Vram
- Intel i5 - 10600K
- 48GB RAM


## Installation
Use the command  ```pip install -r requirements.txt``` to install the required packages.

## Generate Articles


## Train Multi-Granularity Neural Network
Follow the [README.md](code/README.md) provided by Da San Martino et al. or just run ```python train.py --mgn --sig --training --batch_size 16 --lr 3e-5 --alpha 0.9 --n_epochs 20 --patience 7```


## Evaluate Articles

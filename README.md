# REP

This is the code implementation for the paper "REP: A Robustness Enhanced Plugin Method for Differentiable Neural Architecture Search". 

## Requirements

### CNN
- Python 3.6.4
- torch 1.8.0 + cu111
- torchvision 0.9.0 + cu111
- torchattacks 3.4.0

### GNN
- Python 3.7.4
- torch 1.6.0
- torch-cluster 1.5.7
- torch-geometric 1.6.3
- torch-scatter 2.0.6
- torch-sparse 0.6.7
- torch-spline-conv 1.2.0

## Run

### CNN
Run `train_search.py` to perform the search process and run `adv_train.py` to train the searched architecures.

### GNN
Run `train_search.py` to perform the search process and run `train4tune.py` to train the searched architecures.

## References
- [DARTS](https://github.com/quark0/darts)
- [SANE](https://github.com/LARS-research/SANE)
- [AdvRush](https://github.com/nutellamok/advrush/tree/main)
- [torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch)
- [DeepRobust](https://github.com/DSE-MSU/DeepRobust)

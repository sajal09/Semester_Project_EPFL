#!/bin/bash
# Install PyTorch and dependencies first
pip install --upgrade pip
pip install packaging --upgrade
pip install torch==2.8.0
pip install --no-cache-dir  torch_geometric
pip install --no-cache-dir pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://pytorch-geometric.com/whl/torch-2.8.0%2Bcu128.html
pip install --no-cache-dir sgmllib3k
pip install --no-cache-dir pytorch-lightning
pip install --no-cache-dir h5py
pip install --no-cache-dir scikit-learn
pip install --no-cache-dir wandb
pip install --no-cache-dir pytorch-lightning
pip install --no-cache-dir einops
pip install --no-cache-dir rich

cd Task1 

git clone https://github.com/state-spaces/s4
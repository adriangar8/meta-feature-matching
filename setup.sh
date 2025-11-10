#!/bin/bash
conda create -n meta-matching python=3.10 -y
conda activate meta-matching
pip install -r requirements.txt
mkdir -p data results/logs results/checkpoints
wandb login
echo "Setup complete! Ready to train."

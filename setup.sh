#!/bin/bash

read -p "Do you want to create a new conda environment named 'meta-matching'? (y/n): " create_env

if [[ "$create_env" == "y" || "$create_env" == "Y" ]]; then
    conda create -n meta-matching python=3.10 -y
    conda activate meta-matching
else
    echo "Skipping conda environment creation."
fi

pip install -r requirements.txt
mkdir -p data results/logs results/checkpoints
wandb login
echo "Setup complete! Ready to train."
import torch
from src.utils.metrics import compute_matching_accuracy

def evaluate(model, dataloaders):
    results = {}
    for domain, loader in dataloaders.items():
        acc = compute_matching_accuracy(model, loader)
        results[domain] = acc
    return results

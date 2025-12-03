"""
Corrected MAML Evaluation

The key insight about MAML:
- MAML doesn't make the base model better at all tasks
- MAML makes the base model ADAPTABLE - it can quickly become good with few examples
- We need to evaluate: accuracy AFTER few-shot adaptation

Comparison:
- Standard model: Train on A, test on B → bad
- Fine-tuned model: Train on A, fine-tune on B → good on B, forgets A
- MAML model: Train on A+B tasks, adapt to B with few examples → good on B, A unchanged
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from typing import Dict, List, Optional
from copy import deepcopy
from tqdm import tqdm


def evaluate_maml_properly(
    model: nn.Module,
    target_loader: DataLoader,
    source_loader: DataLoader,
    criterion: nn.Module,
    lr_inner: float,
    device: str,
    adaptation_steps: List[int] = [0, 1, 5, 10, 20],
    eval_batches: int = 50,
) -> Dict:
    """
    Properly evaluate MAML by measuring adaptation curve.
    
    For each number of adaptation steps:
    1. Clone the model
    2. Adapt on target domain for N steps
    3. Evaluate on target domain
    4. Also check source domain (should stay roughly same since we cloned)
    
    Returns:
        Dict with:
        - steps: list of step counts
        - target_accuracies: accuracy on target after each step count
        - source_accuracy: accuracy on source (unchanged since we clone)
    """
    model.eval()
    
    # Baseline source accuracy (should stay same since we always clone)
    source_acc = _evaluate_accuracy(model, source_loader, device, eval_batches)
    
    results = {
        "steps": adaptation_steps,
        "target_accuracies": [],
        "source_accuracy": source_acc,
    }
    
    # Save initial state
    initial_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    for n_steps in adaptation_steps:
        # Clone model (important! don't modify original)
        adapted_model = deepcopy(model)
        adapted_model.to(device)
        
        if n_steps > 0:
            # Adapt for n_steps
            adapted_model.train()
            optimizer = torch.optim.SGD(adapted_model.parameters(), lr=lr_inner)
            
            step = 0
            for a, p, n in target_loader:
                if step >= n_steps:
                    break
                
                a, p, n = a.to(device), p.to(device), n.to(device)
                
                da, dp, dn = adapted_model(a), adapted_model(p), adapted_model(n)
                loss = criterion(da, dp, dn)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                step += 1
        
        # Evaluate adapted model on target
        adapted_model.eval()
        target_acc = _evaluate_accuracy(adapted_model, target_loader, device, eval_batches)
        results["target_accuracies"].append(target_acc)
        
        # Clean up
        del adapted_model
        torch.cuda.empty_cache() if device == "cuda" else None
    
    # Verify original model unchanged
    model.load_state_dict(initial_state)
    
    return results


def _evaluate_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    max_batches: int = 50,
) -> float:
    """Compute triplet accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, (a, p, n) in enumerate(loader):
            if i >= max_batches:
                break
            
            a, p, n = a.to(device), p.to(device), n.to(device)
            
            da = model(a)
            dp = model(p)
            dn = model(n)
            
            pos_dist = torch.norm(da - dp, dim=1)
            neg_dist = torch.norm(da - dn, dim=1)
            
            correct += (pos_dist < neg_dist).sum().item()
            total += a.size(0)
    
    return correct / total if total > 0 else 0.0


def compare_adaptation_methods(
    model_standard: nn.Module,
    model_maml: nn.Module,
    target_loader: DataLoader,
    criterion: nn.Module,
    lr: float,
    device: str,
) -> Dict:
    """
    Compare how quickly different models adapt.
    
    Standard model: pretrained on source only
    MAML model: meta-trained on source + target tasks
    
    Both are evaluated on target domain after 0, 1, 5, 10, 20 adaptation steps.
    """
    steps = [0, 1, 5, 10, 20]
    
    print("\nComparing adaptation speed...")
    
    # Standard model adaptation curve
    print("  Evaluating standard model...")
    standard_results = evaluate_maml_properly(
        model_standard, target_loader, target_loader, criterion, lr, device, steps
    )
    
    # MAML model adaptation curve
    print("  Evaluating MAML model...")
    maml_results = evaluate_maml_properly(
        model_maml, target_loader, target_loader, criterion, lr, device, steps
    )
    
    # Print comparison
    print("\n  Adaptation Comparison:")
    print("  Steps | Standard | MAML")
    print("  ------|----------|-----")
    for i, s in enumerate(steps):
        std_acc = standard_results["target_accuracies"][i]
        maml_acc = maml_results["target_accuracies"][i]
        better = "←" if maml_acc > std_acc else ""
        print(f"  {s:5d} | {std_acc:.4f}   | {maml_acc:.4f} {better}")
    
    return {
        "steps": steps,
        "standard": standard_results["target_accuracies"],
        "maml": maml_results["target_accuracies"],
    }


def run_full_maml_experiment(
    source_domain: str,
    target_domain: str,
    get_loader_fn,  # Function to get dataloader
    get_model_fn,   # Function to get model
    epochs_pretrain: int = 5,
    epochs_meta: int = 10,
    lr: float = 1e-4,
    lr_inner: float = 1e-3,
    lr_outer: float = 1e-4,
    batch_size: int = 32,
    device: str = "cuda",
) -> Dict:
    """
    Complete MAML experiment with proper evaluation.
    
    Steps:
    1. Train standard model on source domain
    2. Meta-train MAML model on both domains
    3. Compare adaptation curves on target domain
    """
    from src.meta.maml import MAMLTrainer, create_meta_batch
    
    criterion = nn.TripletMarginLoss(margin=1.0)
    
    source_loader = get_loader_fn(source_domain, batch_size)
    target_loader = get_loader_fn(target_domain, batch_size)
    
    # === Step 1: Train standard model on source ===
    print("\n[1/3] Training standard model on source domain...")
    model_standard = get_model_fn().to(device)
    optimizer = torch.optim.Adam(model_standard.parameters(), lr=lr)
    
    for epoch in range(epochs_pretrain):
        model_standard.train()
        for a, p, n in tqdm(source_loader, desc=f"Standard {epoch+1}/{epochs_pretrain}", leave=False):
            a, p, n = a.to(device), p.to(device), n.to(device)
            loss = criterion(model_standard(a), model_standard(p), model_standard(n))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # === Step 2: Meta-train MAML model ===
    print("\n[2/3] Meta-training MAML model...")
    model_maml = get_model_fn().to(device)
    
    # First, pretrain on source (same as standard)
    optimizer = torch.optim.Adam(model_maml.parameters(), lr=lr)
    for epoch in range(epochs_pretrain):
        model_maml.train()
        for a, p, n in tqdm(source_loader, desc=f"MAML pretrain {epoch+1}/{epochs_pretrain}", leave=False):
            a, p, n = a.to(device), p.to(device), n.to(device)
            loss = criterion(model_maml(a), model_maml(p), model_maml(n))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Then meta-train
    maml = MAMLTrainer(model_maml, lr_inner=lr_inner, lr_outer=lr_outer, device=device)
    
    meta_loaders = {
        source_domain: get_loader_fn(source_domain, 4),
        target_domain: get_loader_fn(target_domain, 4),
    }
    
    for epoch in tqdm(range(epochs_meta), desc="Meta-training"):
        tasks = create_meta_batch(meta_loaders, support_batches=2, query_batches=2)
        if tasks:
            maml.meta_train_step(tasks, criterion)
    
    # === Step 3: Compare adaptation ===
    print("\n[3/3] Comparing adaptation curves...")
    comparison = compare_adaptation_methods(
        model_standard, model_maml, target_loader, criterion, lr_inner, device
    )
    
    # === Summary ===
    print("\n" + "="*50)
    print("MAML EXPERIMENT SUMMARY")
    print("="*50)
    
    # Calculate key metrics
    std_0shot = comparison["standard"][0]
    maml_0shot = comparison["maml"][0]
    std_5shot = comparison["standard"][2]  # index 2 = 5 steps
    maml_5shot = comparison["maml"][2]
    
    print(f"0-shot: Standard={std_0shot:.4f}, MAML={maml_0shot:.4f}")
    print(f"5-shot: Standard={std_5shot:.4f}, MAML={maml_5shot:.4f}")
    print(f"MAML advantage at 5-shot: {(maml_5shot - std_5shot)*100:.2f}%")
    
    return {
        "comparison": comparison,
        "model_standard": model_standard,
        "model_maml": model_maml,
    }


if __name__ == "__main__":
    # Example usage
    from src.models.descriptor_wrapper import get_descriptor_model
    from src.datasets.hpatches_loader import get_triplet_dataloader
    
    results = run_full_maml_experiment(
        source_domain="illumination",
        target_domain="viewpoint",
        get_loader_fn=lambda d, b: get_triplet_dataloader(d, batch_size=b),
        get_model_fn=lambda: get_descriptor_model("resnet50"),
        epochs_pretrain=3,
        epochs_meta=5,
    )
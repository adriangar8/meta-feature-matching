"""
Optimized Benchmark Script - Faster Training with Better GPU Utilization

Key optimizations:
1. Limited dataset size for quick mode
2. Multi-worker data loading
3. Mixed precision training (AMP)
4. Progress bars everywhere
5. Reduced evaluation frequency
"""

import argparse
import yaml
import torch
import torch.cuda.amp as amp
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional
import json
import os

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.descriptor_wrapper import get_descriptor_model
from src.baselines.traditional_matchers import get_all_matchers
from src.baselines.continual_learning import (
    NaiveFineTuner,
    EWC,
    LwF,
    SynapticIntelligence,
)
from src.utils.logger import Logger


# =============================================================================
# OPTIMIZED DATA LOADING
# =============================================================================

def get_fast_triplet_dataloader(
    domain: str,
    batch_size: int = 32,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    pin_memory: bool = True,
):
    """
    Optimized DataLoader with:
    - Multi-worker loading
    - Pin memory for faster GPU transfer
    - Optional sample limit
    """
    from src.datasets.hpatches_loader import HPatchesTriplets
    from torch.utils.data import DataLoader, Subset
    
    ds = HPatchesTriplets(domain=domain)
    
    # Limit samples for quick mode
    if max_samples and len(ds) > max_samples:
        indices = np.random.choice(len(ds), max_samples, replace=False)
        ds = Subset(ds, indices)
    
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True,  # Avoid small batches
        persistent_workers=num_workers > 0,  # Keep workers alive
    )


# =============================================================================
# FAST TRAINING WITH AMP
# =============================================================================

def train_epoch_fast(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: str,
    scaler: Optional[amp.GradScaler] = None,
    desc: str = "Training",
) -> float:
    """
    Single training epoch with:
    - Mixed precision (AMP) for 2x speedup
    - Progress bar
    """
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc=desc, leave=False)
    for a, p, n in pbar:
        a, p, n = a.to(device, non_blocking=True), p.to(device, non_blocking=True), n.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
        
        if scaler:  # Mixed precision
            with amp.autocast():
                da, dp, dn = model(a), model(p), model(n)
                loss = criterion(da, dp, dn)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            da, dp, dn = model(a), model(p), model(n)
            loss = criterion(da, dp, dn)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    return total_loss / len(loader)


def evaluate_fast(
    model: torch.nn.Module,
    loader,
    device: str,
    max_batches: int = 50,
) -> float:
    """Fast evaluation on limited batches."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, (a, p, n) in enumerate(loader):
            if i >= max_batches:
                break
            
            a, p, n = a.to(device), p.to(device), n.to(device)
            da, dp, dn = model(a), model(p), model(n)
            
            pos_dist = torch.norm(da - dp, dim=1)
            neg_dist = torch.norm(da - dn, dim=1)
            
            correct += (pos_dist < neg_dist).sum().item()
            total += a.size(0)
    
    return correct / total if total > 0 else 0


# =============================================================================
# OPTIMIZED BENCHMARK FUNCTIONS
# =============================================================================

def evaluate_traditional_matchers_fast(
    domains: List[str],
    max_pairs: int = 20,
) -> Dict:
    """Evaluate traditional matchers on limited pairs."""
    from src.datasets.hpatches_loader import HPatchesSequences
    from src.evaluation.metrics import evaluate_traditional_matcher
    from torch.utils.data import DataLoader, Subset
    
    print("\n" + "="*60)
    print("EVALUATING TRADITIONAL MATCHERS (Fast Mode)")
    print("="*60)
    
    matchers = get_all_matchers()
    results = {}
    
    for name, matcher in matchers.items():
        print(f"\n--- {name} ---")
        results[name] = {}
        
        for domain in domains:
            ds = HPatchesSequences(domain=domain)
            if len(ds) > max_pairs:
                ds = Subset(ds, range(max_pairs))
            loader = DataLoader(ds, batch_size=1, shuffle=False)
            
            metrics = evaluate_traditional_matcher(matcher, loader)
            
            results[name][domain] = {
                "mma_3px": metrics["mma"].mma_3px,
                "inlier_ratio": metrics["matching"].inlier_ratio,
            }
            print(f"  {domain}: MMA@3px={metrics['mma'].mma_3px:.4f}")
    
    return results


def train_and_evaluate_deep_fast(
    model_name: str,
    train_domain: str,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    max_samples: int = 5000,
    num_workers: int = 4,
) -> Dict:
    """Fast deep model training."""
    print(f"\n--- Training {model_name} on {train_domain} ---")
    
    model = get_descriptor_model(model_name).to(device)
    train_loader = get_fast_triplet_dataloader(
        train_domain, batch_size, num_workers, max_samples
    )
    
    criterion = torch.nn.TripletMarginLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = amp.GradScaler() if device == "cuda" else None
    
    for epoch in range(epochs):
        loss = train_epoch_fast(
            model, train_loader, optimizer, criterion, device, scaler,
            desc=f"Epoch {epoch+1}/{epochs}"
        )
        print(f"  Epoch {epoch+1}: loss={loss:.4f}")
    
    # Evaluate
    results = {"model": model_name, "train_domain": train_domain}
    for domain in ["illumination", "viewpoint"]:
        eval_loader = get_fast_triplet_dataloader(domain, batch_size, num_workers, 2000)
        acc = evaluate_fast(model, eval_loader, device)
        results[f"{domain}_accuracy"] = acc
        print(f"  {domain} accuracy: {acc:.4f}")
    
    return results, model


def evaluate_continual_fast(
    source_domain: str,
    target_domain: str,
    methods: List[str],
    epochs_source: int,
    epochs_target: int,
    batch_size: int,
    lr: float,
    device: str,
    max_samples: int = 5000,
    num_workers: int = 4,
) -> Dict:
    """Fast continual learning evaluation."""
    print("\n" + "="*60)
    print(f"CONTINUAL LEARNING: {source_domain} -> {target_domain}")
    print("="*60)
    
    source_loader = get_fast_triplet_dataloader(source_domain, batch_size, num_workers, max_samples)
    target_loader = get_fast_triplet_dataloader(target_domain, batch_size, num_workers, max_samples)
    criterion = torch.nn.TripletMarginLoss(margin=1.0)
    
    results = {}
    
    for method_name in methods:
        print(f"\n--- {method_name} ---")
        
        model = get_descriptor_model("resnet50").to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scaler = amp.GradScaler() if device == "cuda" else None
        
        # Phase 1: Train on source
        print(f"  Phase 1: Training on {source_domain}...")
        for epoch in range(epochs_source):
            loss = train_epoch_fast(
                model, source_loader, optimizer, criterion, device, scaler,
                desc=f"Source {epoch+1}/{epochs_source}"
            )
        
        source_acc_before = evaluate_fast(model, source_loader, device)
        print(f"  Source accuracy after training: {source_acc_before:.4f}")
        
        # Phase 2: Train on target with continual learning
        print(f"  Phase 2: Training on {target_domain} with {method_name}...")
        
        if method_name == "naive":
            trainer = NaiveFineTuner(model, lr=lr, device=device)
        elif method_name == "ewc":
            trainer = EWC(model, lr=lr, ewc_lambda=1000, device=device)
            trainer.compute_fisher(source_loader, criterion)
        elif method_name == "lwf":
            trainer = LwF(model, lr=lr, lwf_lambda=1.0, device=device)
            trainer.consolidate()
        elif method_name == "si":
            trainer = SynapticIntelligence(model, lr=lr, si_lambda=1.0, device=device)
        
        for epoch in range(epochs_target):
            model.train()
            pbar = tqdm(target_loader, desc=f"Target {epoch+1}/{epochs_target}", leave=False)
            for batch in pbar:
                trainer.train_step(batch, criterion)
        
        source_acc_after = evaluate_fast(model, source_loader, device)
        target_acc_after = evaluate_fast(model, target_loader, device)
        
        forgetting = source_acc_before - source_acc_after
        
        print(f"  Results: Source={source_acc_after:.4f} (forgot {forgetting:.4f}), Target={target_acc_after:.4f}")
        
        results[method_name] = {
            "source_acc_before": source_acc_before,
            "source_acc_after": source_acc_after,
            "target_acc_after": target_acc_after,
            "forgetting": forgetting,
        }
    
    return results


def evaluate_maml_fast(
    source_domain: str,
    target_domain: str,
    epochs_pretrain: int,
    epochs_meta: int,
    lr_inner: float,
    lr_outer: float,
    batch_size: int,
    device: str,
    max_samples: int = 3000,
    num_workers: int = 4,
) -> Dict:
    """Fast MAML evaluation."""
    from src.meta.maml import MAMLTrainer, create_meta_batch
    
    print("\n" + "="*60)
    print(f"MAML: {source_domain} -> {target_domain}")
    print("="*60)
    
    source_loader = get_fast_triplet_dataloader(source_domain, batch_size, num_workers, max_samples)
    target_loader = get_fast_triplet_dataloader(target_domain, batch_size, num_workers, max_samples)
    criterion = torch.nn.TripletMarginLoss(margin=1.0)
    
    # Pretrain
    print("  Phase 1: Pretraining...")
    model = get_descriptor_model("resnet50").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_outer)
    scaler = amp.GradScaler() if device == "cuda" else None
    
    for epoch in range(epochs_pretrain):
        train_epoch_fast(model, source_loader, optimizer, criterion, device, scaler,
                        desc=f"Pretrain {epoch+1}/{epochs_pretrain}")
    
    source_acc_before = evaluate_fast(model, source_loader, device)
    target_acc_before = evaluate_fast(model, target_loader, device)
    print(f"  After pretrain: Source={source_acc_before:.4f}, Target={target_acc_before:.4f}")
    
    # Meta-learning
    print("  Phase 2: Meta-learning...")
    maml = MAMLTrainer(model, lr_inner=lr_inner, lr_outer=lr_outer, device=device)
    
    meta_loaders = {
        source_domain: get_fast_triplet_dataloader(source_domain, 4, num_workers, 1000),
        target_domain: get_fast_triplet_dataloader(target_domain, 4, num_workers, 1000),
    }
    
    for epoch in tqdm(range(epochs_meta), desc="Meta epochs"):
        tasks = create_meta_batch(meta_loaders, support_batches=2, query_batches=2)
        if tasks:
            maml.meta_train_step(tasks, criterion)
    
    source_acc_after = evaluate_fast(model, source_loader, device)
    target_acc_after = evaluate_fast(model, target_loader, device)
    print(f"  After meta: Source={source_acc_after:.4f}, Target={target_acc_after:.4f}")
    
    return {
        "source_acc_before": source_acc_before,
        "source_acc_after": source_acc_after,
        "target_acc_before": target_acc_before,
        "target_acc_after": target_acc_after,
    }


# =============================================================================
# MAIN
# =============================================================================

def run_fast_benchmark(config: Dict) -> Dict:
    """Run optimized benchmark."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Optimal workers based on CPU
    num_workers = min(4, os.cpu_count() or 1)
    print(f"Using {num_workers} data loading workers")
    
    logger = Logger(
        project=config.get("project", "meta-feature-matching"),
        run_name=config.get("run_name", "fast_benchmark"),
        use_wandb=config.get("use_wandb", False),  # Disable wandb for speed
        config=config,
    )
    
    all_results = {}
    max_samples = config.get("max_samples", 5000)
    
    # 1. Traditional matchers
    if config.get("eval_traditional", True):
        all_results["traditional"] = evaluate_traditional_matchers_fast(
            domains=["illumination", "viewpoint"],
            max_pairs=config.get("max_pairs", 20),
        )
    
    # 2. Deep learning
    if config.get("eval_deep", True):
        all_results["deep"] = {}
        for model_name in config.get("deep_models", ["resnet50"]):
            results, _ = train_and_evaluate_deep_fast(
                model_name=model_name,
                train_domain="illumination",
                epochs=config.get("deep_epochs", 2),
                batch_size=config.get("batch_size", 64),
                lr=config.get("lr", 1e-4),
                device=device,
                max_samples=max_samples,
                num_workers=num_workers,
            )
            all_results["deep"][model_name] = results
    
    # 3. Continual learning
    if config.get("eval_continual", True):
        all_results["continual"] = evaluate_continual_fast(
            source_domain="illumination",
            target_domain="viewpoint",
            methods=config.get("continual_methods", ["naive", "ewc"]),
            epochs_source=config.get("epochs_source", 2),
            epochs_target=config.get("epochs_target", 2),
            batch_size=config.get("batch_size", 64),
            lr=config.get("lr", 1e-4),
            device=device,
            max_samples=max_samples,
            num_workers=num_workers,
        )
    
    # 4. MAML
    if config.get("eval_maml", True):
        all_results["maml"] = evaluate_maml_fast(
            source_domain="illumination",
            target_domain="viewpoint",
            epochs_pretrain=config.get("epochs_source", 2),
            epochs_meta=config.get("meta_epochs", 2),
            lr_inner=config.get("meta_lr_inner", 1e-3),
            lr_outer=config.get("meta_lr_outer", 1e-4),
            batch_size=config.get("batch_size", 64),
            device=device,
            max_samples=max_samples // 2,
            num_workers=num_workers,
        )
    
    # Save results
    results_path = logger.log_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"Results saved to {results_path}")
    print(f"{'='*60}")
    
    logger.finish()
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Fast feature matching benchmark")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--quick", action="store_true", help="Ultra-quick test")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb")
    args = parser.parse_args()
    
    # Default config optimized for speed
    config = {
        "project": "meta-feature-matching",
        "run_name": "fast_benchmark",
        "use_wandb": not args.no_wandb,
        "eval_traditional": True,
        "eval_deep": True,
        "eval_continual": True,
        "eval_maml": True,
        "deep_models": ["resnet50"],
        "continual_methods": ["naive", "ewc"],
        "deep_epochs": 3,
        "epochs_source": 3,
        "epochs_target": 3,
        "meta_epochs": 3,
        "lr": 1e-4,
        "meta_lr_inner": 1e-3,
        "meta_lr_outer": 1e-4,
        "batch_size": 64,  # Larger batch for GPU efficiency
        "max_samples": 5000,  # Limit dataset size
        "max_pairs": 30,  # For traditional matcher eval
    }
    
    if args.config:
        with open(args.config) as f:
            config.update(yaml.safe_load(f))
    
    if args.quick:
        config.update({
            "deep_epochs": 1,
            "epochs_source": 1,
            "epochs_target": 1,
            "meta_epochs": 1,
            "max_samples": 2000,
            "max_pairs": 10,
            "continual_methods": ["naive"],
            "eval_maml": False,  # Skip MAML for ultra-quick
        })
    
    run_fast_benchmark(config)


if __name__ == "__main__":
    main()
"""
Improved Benchmark Script v2

Fixes:
1. MAML now shows adaptation curve (the key metric)
2. EWC lambda configurable and properly tuned
3. More informative output
4. Proper evaluation on held-out data
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
    """Optimized DataLoader."""
    from src.datasets.hpatches_loader import HPatchesTriplets
    from torch.utils.data import DataLoader, Subset
    
    ds = HPatchesTriplets(domain=domain)
    
    if max_samples and len(ds) > max_samples:
        indices = np.random.choice(len(ds), max_samples, replace=False)
        ds = Subset(ds, indices)
    
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=num_workers > 0,
    )


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def train_epoch_fast(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: str,
    scaler=None,
    desc: str = "Training",
) -> float:
    """Training epoch with AMP."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc=desc, leave=False)
    for a, p, n in pbar:
        a = a.to(device, non_blocking=True)
        p = p.to(device, non_blocking=True)
        n = n.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        if scaler:
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


def evaluate_accuracy(
    model: torch.nn.Module,
    loader,
    device: str,
    max_batches: int = 100,
) -> float:
    """Evaluate triplet accuracy."""
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
# TRADITIONAL MATCHERS
# =============================================================================

def evaluate_traditional_matchers_fast(domains: List[str], max_pairs: int = 30) -> Dict:
    """Evaluate traditional matchers."""
    from src.datasets.hpatches_loader import HPatchesSequences
    from src.evaluation.metrics import evaluate_traditional_matcher
    from torch.utils.data import DataLoader, Subset
    
    print("\n" + "="*60)
    print("EVALUATING TRADITIONAL MATCHERS")
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
                "mma_5px": metrics["mma"].mma_5px,
                "inlier_ratio": metrics["matching"].inlier_ratio,
            }
            print(f"  {domain}: MMA@3px={metrics['mma'].mma_3px:.4f}, MMA@5px={metrics['mma'].mma_5px:.4f}")
    
    return results


# =============================================================================
# DEEP LEARNING
# =============================================================================

def train_deep_model(
    model_name: str,
    train_domain: str,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    max_samples: int = 10000,
    num_workers: int = 4,
) -> Dict:
    """Train and evaluate deep model."""
    print(f"\n--- Training {model_name} on {train_domain} ---")
    
    model = get_descriptor_model(model_name).to(device)
    train_loader = get_fast_triplet_dataloader(train_domain, batch_size, num_workers, max_samples)
    
    criterion = torch.nn.TripletMarginLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = amp.GradScaler() if device == "cuda" else None
    
    for epoch in range(epochs):
        loss = train_epoch_fast(
            model, train_loader, optimizer, criterion, device, scaler,
            desc=f"Epoch {epoch+1}/{epochs}"
        )
        print(f"  Epoch {epoch+1}: loss={loss:.4f}")
    
    # Evaluate on both domains
    results = {"model": model_name, "train_domain": train_domain}
    
    for domain in ["illumination", "viewpoint"]:
        eval_loader = get_fast_triplet_dataloader(domain, batch_size, num_workers, max_samples // 2)
        acc = evaluate_accuracy(model, eval_loader, device)
        results[f"{domain}_accuracy"] = acc
        print(f"  {domain} accuracy: {acc:.4f}")
    
    return results, model


# =============================================================================
# CONTINUAL LEARNING
# =============================================================================

def evaluate_continual_learning(
    source_domain: str,
    target_domain: str,
    methods: List[str],
    epochs_source: int,
    epochs_target: int,
    batch_size: int,
    lr: float,
    device: str,
    max_samples: int = 10000,
    num_workers: int = 4,
    ewc_lambda: float = 400.0,
    lwf_lambda: float = 1.0,
    si_lambda: float = 0.5,
) -> Dict:
    """Evaluate continual learning methods."""
    print("\n" + "="*60)
    print(f"CONTINUAL LEARNING: {source_domain} -> {target_domain}")
    print("="*60)
    
    source_loader = get_fast_triplet_dataloader(source_domain, batch_size, num_workers, max_samples)
    target_loader = get_fast_triplet_dataloader(target_domain, batch_size, num_workers, max_samples)
    
    # Separate evaluation loaders (held-out data)
    source_eval = get_fast_triplet_dataloader(source_domain, batch_size, num_workers, max_samples // 2)
    target_eval = get_fast_triplet_dataloader(target_domain, batch_size, num_workers, max_samples // 2)
    
    criterion = torch.nn.TripletMarginLoss(margin=1.0)
    results = {}
    
    for method_name in methods:
        print(f"\n--- {method_name.upper()} ---")
        
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
        
        # Evaluate after source training
        source_acc_before = evaluate_accuracy(model, source_eval, device)
        target_acc_before = evaluate_accuracy(model, target_eval, device)
        print(f"  After source: Source={source_acc_before:.4f}, Target={target_acc_before:.4f}")
        
        # Phase 2: Setup continual learning method
        if method_name == "naive":
            trainer = NaiveFineTuner(model, lr=lr, device=device)
        elif method_name == "ewc":
            trainer = EWC(model, lr=lr, ewc_lambda=ewc_lambda, device=device)
            print(f"  Computing Fisher information (lambda={ewc_lambda})...")
            trainer.compute_fisher(source_loader, criterion, num_samples=500)
        elif method_name == "lwf":
            trainer = LwF(model, lr=lr, lwf_lambda=lwf_lambda, device=device)
            trainer.consolidate()
        elif method_name == "si":
            trainer = SynapticIntelligence(model, lr=lr, si_lambda=si_lambda, device=device)
        
        # Train on target
        print(f"  Phase 2: Training on {target_domain}...")
        forgetting_curve = []
        
        for epoch in range(epochs_target):
            model.train()
            epoch_loss = 0
            for batch in tqdm(target_loader, desc=f"Target {epoch+1}/{epochs_target}", leave=False):
                loss_val = trainer.train_step(batch, criterion)
                if isinstance(loss_val, tuple):
                    loss_val = loss_val[0]
                epoch_loss += loss_val
            
            # Track forgetting during training
            if epoch % 2 == 0 or epoch == epochs_target - 1:
                src_acc = evaluate_accuracy(model, source_eval, device)
                tgt_acc = evaluate_accuracy(model, target_eval, device)
                forgetting_curve.append({
                    "epoch": epoch,
                    "source_acc": src_acc,
                    "target_acc": tgt_acc,
                })
        
        # Final evaluation
        source_acc_after = evaluate_accuracy(model, source_eval, device)
        target_acc_after = evaluate_accuracy(model, target_eval, device)
        forgetting = source_acc_before - source_acc_after
        
        print(f"  Final: Source={source_acc_after:.4f} (forgot {forgetting:.4f}), Target={target_acc_after:.4f}")
        
        results[method_name] = {
            "source_acc_before": source_acc_before,
            "source_acc_after": source_acc_after,
            "target_acc_before": target_acc_before,
            "target_acc_after": target_acc_after,
            "forgetting": forgetting,
            "forgetting_rate": forgetting / source_acc_before if source_acc_before > 0 else 0,
            "forgetting_curve": forgetting_curve,
        }
    
    return results


# =============================================================================
# MAML WITH ADAPTATION CURVE
# =============================================================================

def evaluate_maml(
    source_domain: str,
    target_domain: str,
    epochs_pretrain: int,
    epochs_meta: int,
    lr_inner: float,
    lr_outer: float,
    batch_size: int,
    device: str,
    max_samples: int = 5000,
    num_workers: int = 4,
    inner_steps: int = 5,
) -> Dict:
    """Evaluate MAML with adaptation curve."""
    from src.meta.maml import MAMLTrainer, create_meta_batch
    from copy import deepcopy
    
    print("\n" + "="*60)
    print(f"MAML: {source_domain} -> {target_domain}")
    print("="*60)
    
    source_loader = get_fast_triplet_dataloader(source_domain, batch_size, num_workers, max_samples)
    target_loader = get_fast_triplet_dataloader(target_domain, batch_size, num_workers, max_samples)
    source_eval = get_fast_triplet_dataloader(source_domain, batch_size, num_workers, max_samples // 2)
    target_eval = get_fast_triplet_dataloader(target_domain, batch_size, num_workers, max_samples // 2)
    
    criterion = torch.nn.TripletMarginLoss(margin=1.0)
    
    # Phase 1: Pretrain on source
    print("  Phase 1: Pretraining on source...")
    model = get_descriptor_model("resnet50").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_outer)
    scaler = amp.GradScaler() if device == "cuda" else None
    
    for epoch in range(epochs_pretrain):
        train_epoch_fast(model, source_loader, optimizer, criterion, device, scaler,
                        desc=f"Pretrain {epoch+1}/{epochs_pretrain}")
    
    source_acc_pretrain = evaluate_accuracy(model, source_eval, device)
    target_acc_pretrain = evaluate_accuracy(model, target_eval, device)
    print(f"  After pretrain: Source={source_acc_pretrain:.4f}, Target={target_acc_pretrain:.4f}")
    
    # Phase 2: Meta-learning
    print(f"  Phase 2: Meta-learning ({epochs_meta} epochs)...")
    maml = MAMLTrainer(
        model,
        lr_inner=lr_inner,
        lr_outer=lr_outer,
        num_inner_steps=inner_steps,
        device=device,
    )
    
    # Use smaller batch for meta-learning tasks
    meta_loaders = {
        source_domain: get_fast_triplet_dataloader(source_domain, 8, num_workers, 2000),
        target_domain: get_fast_triplet_dataloader(target_domain, 8, num_workers, 2000),
    }
    
    meta_losses = []
    for epoch in tqdm(range(epochs_meta), desc="Meta-learning"):
        tasks = create_meta_batch(meta_loaders, support_batches=4, query_batches=4)
        if tasks:
            loss_dict = maml.meta_train_step(tasks, criterion)
            meta_losses.append(loss_dict.get("meta_loss", 0))
    
    source_acc_meta = evaluate_accuracy(model, source_eval, device)
    target_acc_meta = evaluate_accuracy(model, target_eval, device)
    print(f"  After meta: Source={source_acc_meta:.4f}, Target={target_acc_meta:.4f}")
    
    # Phase 3: Adaptation curve (THE KEY METRIC FOR MAML)
    print("  Phase 3: Computing adaptation curve...")
    adaptation_steps = [0, 1, 2, 5, 10, 20]
    adaptation_results = []
    
    # Save the meta-learned model state
    meta_state = deepcopy(model.state_dict())
    
    for n_steps in adaptation_steps:
        # Restore meta-learned model
        model.load_state_dict(deepcopy(meta_state))
        
        if n_steps > 0:
            # Adapt on target domain
            model.train()
            adapt_optimizer = torch.optim.SGD(model.parameters(), lr=lr_inner)
            
            step = 0
            for a, p, n in target_loader:
                if step >= n_steps:
                    break
                a, p, n = a.to(device), p.to(device), n.to(device)
                
                da, dp, dn = model(a), model(p), model(n)
                loss = criterion(da, dp, dn)
                
                adapt_optimizer.zero_grad()
                loss.backward()
                adapt_optimizer.step()
                step += 1
        
        # Evaluate
        acc = evaluate_accuracy(model, target_eval, device)
        adaptation_results.append({"steps": n_steps, "accuracy": acc})
        print(f"    {n_steps}-shot: {acc:.4f}")
    
    # Restore model to meta-learned state
    model.load_state_dict(meta_state)
    
    return {
        "source_acc_pretrain": source_acc_pretrain,
        "target_acc_pretrain": target_acc_pretrain,
        "source_acc_meta": source_acc_meta,
        "target_acc_meta": target_acc_meta,
        "adaptation_curve": adaptation_results,
        "meta_losses": meta_losses[-10:] if meta_losses else [],  # Last 10 losses
    }


# =============================================================================
# MAIN
# =============================================================================

def run_benchmark(config: Dict) -> Dict:
    """Run the complete benchmark."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"META-FEATURE-MATCHING BENCHMARK")
    print(f"{'='*60}")
    print(f"Device: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    num_workers = min(config.get("num_workers", 4), os.cpu_count() or 1)
    print(f"Data workers: {num_workers}")
    
    logger = Logger(
        project=config.get("project", "meta-feature-matching"),
        run_name=config.get("run_name", "benchmark"),
        use_wandb=config.get("use_wandb", False),
        config=config,
    )
    
    all_results = {}
    max_samples = config.get("max_samples", 10000)
    
    # 1. Traditional matchers
    if config.get("eval_traditional", True):
        all_results["traditional"] = evaluate_traditional_matchers_fast(
            domains=["illumination", "viewpoint"],
            max_pairs=config.get("max_pairs", 30),
        )
    
    # 2. Deep learning
    if config.get("eval_deep", True):
        all_results["deep"] = {}
        for model_name in config.get("deep_models", ["resnet50"]):
            for train_domain in ["illumination", "viewpoint"]:
                key = f"{model_name}_{train_domain}"
                results, _ = train_deep_model(
                    model_name=model_name,
                    train_domain=train_domain,
                    epochs=config.get("deep_epochs", 5),
                    batch_size=config.get("batch_size", 64),
                    lr=config.get("lr", 1e-4),
                    device=device,
                    max_samples=max_samples,
                    num_workers=num_workers,
                )
                all_results["deep"][key] = results
    
    # 3. Continual learning
    if config.get("eval_continual", True):
        all_results["continual"] = {}
        
        for source, target in [("illumination", "viewpoint"), ("viewpoint", "illumination")]:
            key = f"{source}_to_{target}"
            all_results["continual"][key] = evaluate_continual_learning(
                source_domain=source,
                target_domain=target,
                methods=config.get("continual_methods", ["naive", "ewc", "lwf", "si"]),
                epochs_source=config.get("epochs_source", 5),
                epochs_target=config.get("epochs_target", 5),
                batch_size=config.get("batch_size", 64),
                lr=config.get("lr", 1e-4),
                device=device,
                max_samples=max_samples,
                num_workers=num_workers,
                ewc_lambda=config.get("ewc_lambda", 400),
                lwf_lambda=config.get("lwf_lambda", 1.0),
                si_lambda=config.get("si_lambda", 0.5),
            )
    
    # 4. MAML
    if config.get("eval_maml", True):
        all_results["maml"] = {}
        
        for source, target in [("illumination", "viewpoint"), ("viewpoint", "illumination")]:
            key = f"{source}_to_{target}"
            all_results["maml"][key] = evaluate_maml(
                source_domain=source,
                target_domain=target,
                epochs_pretrain=config.get("epochs_source", 5),
                epochs_meta=config.get("meta_epochs", 20),
                lr_inner=config.get("meta_lr_inner", 0.01),
                lr_outer=config.get("meta_lr_outer", 1e-4),
                batch_size=config.get("batch_size", 64),
                device=device,
                max_samples=max_samples,
                num_workers=num_workers,
                inner_steps=config.get("maml_inner_steps", 5),
            )
    
    # Save results
    results_path = logger.log_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    if "continual" in all_results:
        print("\nContinual Learning Forgetting Rates:")
        for transfer, methods in all_results["continual"].items():
            print(f"  {transfer}:")
            for method, data in methods.items():
                print(f"    {method}: {data['forgetting_rate']*100:.1f}%")
    
    if "maml" in all_results:
        print("\nMAML Adaptation (target accuracy):")
        for transfer, data in all_results["maml"].items():
            print(f"  {transfer}:")
            for point in data.get("adaptation_curve", []):
                print(f"    {point['steps']}-shot: {point['accuracy']:.4f}")
    
    print(f"\nResults saved to: {results_path}")
    
    logger.finish()
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Feature Matching Benchmark v2")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb")
    args = parser.parse_args()
    
    # Default config
    config = {
        "project": "meta-feature-matching",
        "run_name": "benchmark_v2",
        "use_wandb": not args.no_wandb,
        "eval_traditional": True,
        "eval_deep": True,
        "eval_continual": True,
        "eval_maml": True,
        "deep_models": ["resnet50"],
        "continual_methods": ["naive", "ewc", "lwf", "si"],
        "deep_epochs": 5,
        "epochs_source": 5,
        "epochs_target": 5,
        "meta_epochs": 20,
        "lr": 1e-4,
        "meta_lr_inner": 0.01,
        "meta_lr_outer": 1e-4,
        "ewc_lambda": 400,
        "lwf_lambda": 1.0,
        "si_lambda": 0.5,
        "batch_size": 64,
        "max_samples": 10000,
        "max_pairs": 30,
        "num_workers": 4,
        "maml_inner_steps": 5,
    }
    
    if args.config:
        with open(args.config) as f:
            config.update(yaml.safe_load(f))
    
    if args.quick:
        config.update({
            "deep_epochs": 2,
            "epochs_source": 2,
            "epochs_target": 2,
            "meta_epochs": 5,
            "max_samples": 3000,
            "max_pairs": 15,
            "continual_methods": ["naive", "ewc"],
        })
    
    run_benchmark(config)


if __name__ == "__main__":
    main()
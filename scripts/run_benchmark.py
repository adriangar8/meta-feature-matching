"""
Comprehensive Benchmark Script

Compares:
1. Traditional matchers: SIFT, ORB, BRISK, AKAZE (with BF and FLANN)
2. Deep learning descriptors: ResNet, Lightweight, HardNet
3. Continual learning methods: Naive, EWC, LwF, SI, MAML

Evaluates on:
- Illumination domain
- Viewpoint domain
- Cross-domain transfer

Metrics:
- Triplet accuracy
- MMA at various thresholds
- Homography estimation accuracy
- Forgetting rates
"""

import argparse
import yaml
import torch
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List
import json

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets.hpatches_loader import (
    get_triplet_dataloader,
    get_sequence_dataloader,
)
from src.models.descriptor_wrapper import get_descriptor_model
from src.baselines.traditional_matchers import get_all_matchers
from src.baselines.deep_matcher import DeepFeatureMatcher
from src.baselines.continual_learning import (
    NaiveFineTuner,
    EWC,
    LwF,
    SynapticIntelligence,
)
from src.meta.maml import MAMLTrainer, create_meta_batch
from src.evaluation.metrics import (
    triplet_accuracy,
    evaluate_traditional_matcher,
    evaluate_deep_matcher,
    compute_forgetting,
)
from src.evaluation.visualization import (
    plot_method_comparison,
    plot_mma_comparison,
    plot_forgetting_curve,
    plot_domain_comparison,
    create_summary_table,
)
from src.utils.logger import Logger


def evaluate_traditional_matchers(
    domains: List[str],
    logger: Logger,
) -> Dict[str, Dict]:
    """Evaluate all traditional matchers."""
    print("\n" + "="*60)
    print("EVALUATING TRADITIONAL MATCHERS")
    print("="*60)

    matchers = get_all_matchers()
    results = {}

    for name, matcher in matchers.items():
        print(f"\n--- {name} ---")
        results[name] = {}

        for domain in domains:
            print(f"  Domain: {domain}")
            seq_loader = get_sequence_dataloader(domain)
            
            metrics = evaluate_traditional_matcher(matcher, seq_loader)
            
            results[name][domain] = {
                "num_matches": metrics["matching"].num_matches,
                "num_inliers": metrics["matching"].num_inliers,
                "inlier_ratio": metrics["matching"].inlier_ratio,
                "match_time_ms": metrics["matching"].match_time_ms,
                "mma_1px": metrics["mma"].mma_1px,
                "mma_3px": metrics["mma"].mma_3px,
                "mma_5px": metrics["mma"].mma_5px,
                "mma_10px": metrics["mma"].mma_10px,
            }

            if metrics["homography"]:
                results[name][domain].update({
                    "h_correct_1px": metrics["homography"].correctness_1px,
                    "h_correct_3px": metrics["homography"].correctness_3px,
                    "h_correct_5px": metrics["homography"].correctness_5px,
                })

            print(f"    MMA@3px: {metrics['mma'].mma_3px:.4f}")
            print(f"    Inlier ratio: {metrics['matching'].inlier_ratio:.4f}")

    return results


def train_and_evaluate_deep_model(
    model_name: str,
    train_domain: str,
    eval_domains: List[str],
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    logger: Logger,
) -> Dict:
    """Train and evaluate a deep learning descriptor model."""
    print(f"\n--- Training {model_name} on {train_domain} ---")

    model = get_descriptor_model(model_name).to(device)
    train_loader = get_triplet_dataloader(train_domain, batch_size=batch_size)
    criterion = torch.nn.TripletMarginLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scaler = GradScaler()

    # Training
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for a, p, n in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            a, p, n = a.to(device), p.to(device), n.to(device)
            
            optimizer.zero_grad(set_to_none=True)  # Faster
            
            with autocast():  # ADD: Mixed precision
                da, dp, dn = model(a), model(p), model(n)
                loss = criterion(da, dp, dn)
            
            scaler.scale(loss).backward()  # ADD
            scaler.step(optimizer)         # ADD
            scaler.update()                # ADD
            
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logger.log_metrics({"loss": avg_loss}, epoch, prefix=f"deep/{model_name}")

    # Evaluation
    results = {"model": model_name, "train_domain": train_domain}

    for domain in eval_domains:
        eval_loader = get_triplet_dataloader(domain, batch_size=batch_size)
        metrics = triplet_accuracy(model, eval_loader, device)

        results[f"{domain}_accuracy"] = metrics.accuracy
        results[f"{domain}_margin_violation"] = metrics.margin_violation_rate

        print(f"  {domain} accuracy: {metrics.accuracy:.4f}")

    return results, model

def evaluate_continual_learning(
    source_domain: str,
    target_domain: str,
    methods: List[str],
    epochs_source: int,
    epochs_target: int,
    lr: float,
    batch_size: int,
    device: str,
    logger: Logger,
) -> Dict[str, Dict]:
    """
    Evaluate continual learning methods on domain shift.
    
    1. Train on source domain
    2. Record accuracy on source
    3. Train on target domain (with continual learning)
    4. Record accuracy on both domains
    5. Measure forgetting
    """
    print("\n" + "="*60)
    print(f"CONTINUAL LEARNING: {source_domain} -> {target_domain}")
    print("="*60)

    source_loader = get_triplet_dataloader(source_domain, batch_size=batch_size)
    target_loader = get_triplet_dataloader(target_domain, batch_size=batch_size)
    criterion = torch.nn.TripletMarginLoss(margin=1.0)

    results = {}

    for method_name in methods:
        print(f"\n--- {method_name} ---")

        # Fresh model
        model = get_descriptor_model("resnet50").to(device)

        # Phase 1: Train on source domain
        print(f"  Phase 1: Training on {source_domain}...")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs_source):
            model.train()
            for a, p, n in source_loader:
                a, p, n = a.to(device), p.to(device), n.to(device)
                loss = criterion(model(a), model(p), model(n))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Evaluate after source training
        source_acc_before = triplet_accuracy(model, source_loader, device).accuracy
        target_acc_before = triplet_accuracy(model, target_loader, device).accuracy

        print(f"  After source training:")
        print(f"    Source accuracy: {source_acc_before:.4f}")
        print(f"    Target accuracy: {target_acc_before:.4f}")

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

        # Track forgetting during training
        forgetting_curve = {"epochs": [], "source_acc": [], "target_acc": []}

        for epoch in range(epochs_target):
            trainer.train_epoch(target_loader, criterion)

            # Evaluate periodically
            if epoch % 2 == 0 or epoch == epochs_target - 1:
                src_acc = triplet_accuracy(model, source_loader, device).accuracy
                tgt_acc = triplet_accuracy(model, target_loader, device).accuracy

                forgetting_curve["epochs"].append(epoch)
                forgetting_curve["source_acc"].append(src_acc)
                forgetting_curve["target_acc"].append(tgt_acc)

        # Final evaluation
        source_acc_after = triplet_accuracy(model, source_loader, device).accuracy
        target_acc_after = triplet_accuracy(model, target_loader, device).accuracy

        forgetting = compute_forgetting(source_acc_before, source_acc_after)

        print(f"  After target training:")
        print(f"    Source accuracy: {source_acc_after:.4f} (forgetting: {forgetting.forgetting:.4f})")
        print(f"    Target accuracy: {target_acc_after:.4f}")

        results[method_name] = {
            "source_acc_before": source_acc_before,
            "source_acc_after": source_acc_after,
            "target_acc_before": target_acc_before,
            "target_acc_after": target_acc_after,
            "forgetting": forgetting.forgetting,
            "forgetting_rate": forgetting.forgetting_rate,
            "forgetting_curve": forgetting_curve,
        }

    return results


def evaluate_maml(
    source_domain: str,
    target_domain: str,
    epochs_pretrain: int,
    epochs_meta: int,
    lr_inner: float,
    lr_outer: float,
    batch_size: int,
    device: str,
    logger: Logger,
) -> Dict:
    """Evaluate MAML for domain adaptation."""
    print("\n" + "="*60)
    print(f"MAML: {source_domain} -> {target_domain}")
    print("="*60)

    source_loader = get_triplet_dataloader(source_domain, batch_size=batch_size)
    target_loader = get_triplet_dataloader(target_domain, batch_size=batch_size)
    criterion = torch.nn.TripletMarginLoss(margin=1.0)

    # Phase 1: Pretrain on source
    print("  Phase 1: Pretraining on source domain...")
    model = get_descriptor_model("resnet50").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_outer)

    for epoch in range(epochs_pretrain):
        model.train()
        for a, p, n in source_loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            loss = criterion(model(a), model(p), model(n))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    source_acc_before = triplet_accuracy(model, source_loader, device).accuracy
    target_acc_before = triplet_accuracy(model, target_loader, device).accuracy

    print(f"  After pretraining:")
    print(f"    Source accuracy: {source_acc_before:.4f}")
    print(f"    Target accuracy: {target_acc_before:.4f}")

    # Phase 2: Meta-learning
    print("  Phase 2: Meta-learning...")
    maml = MAMLTrainer(
        model,
        lr_inner=lr_inner,
        lr_outer=lr_outer,
        num_inner_steps=3,
        device=device,
    )

    loaders = {
        source_domain: get_triplet_dataloader(source_domain, batch_size=4),
        target_domain: get_triplet_dataloader(target_domain, batch_size=4),
    }

    for epoch in range(epochs_meta):
        tasks = create_meta_batch(loaders, support_batches=2, query_batches=2)
        losses = maml.meta_train_step(tasks, criterion)
        logger.log_metrics(losses, epoch, prefix="maml")

    # Phase 3: Evaluate adaptation
    print("  Phase 3: Evaluating adaptation...")

    # Adaptation curve
    from src.meta.maml import meta_adaptation_curve
    adapt_results = meta_adaptation_curve(
        model, target_loader, criterion, lr_inner, device,
        steps=[0, 1, 2, 5, 10, 20],
    )

    source_acc_after = triplet_accuracy(model, source_loader, device).accuracy
    target_acc_after = triplet_accuracy(model, target_loader, device).accuracy

    print(f"  After meta-learning:")
    print(f"    Source accuracy: {source_acc_after:.4f}")
    print(f"    Target accuracy (0-shot): {adapt_results['accuracies'][0]:.4f}")
    print(f"    Target accuracy (5-shot): {adapt_results['accuracies'][3]:.4f}")

    return {
        "source_acc_before": source_acc_before,
        "source_acc_after": source_acc_after,
        "target_acc_before": target_acc_before,
        "target_acc_after": target_acc_after,
        "adaptation_curve": adapt_results,
    }


def run_full_benchmark(config: Dict) -> Dict:
    """Run the complete benchmark suite."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize logger
    logger = Logger(
        project=config.get("project", "meta-feature-matching"),
        run_name=config.get("run_name", "full_benchmark"),
        use_wandb=config.get("use_wandb", True),
        config=config,
    )

    all_results = {}

    # 1. Traditional matchers
    if config.get("eval_traditional", True):
        all_results["traditional"] = evaluate_traditional_matchers(
            domains=["illumination", "viewpoint"],
            logger=logger,
        )

    # 2. Deep learning baselines
    if config.get("eval_deep", True):
        all_results["deep"] = {}

        for model_name in config.get("deep_models", ["resnet50"]):
            for train_domain in ["illumination", "viewpoint"]:
                results, _ = train_and_evaluate_deep_model(
                    model_name=model_name,
                    train_domain=train_domain,
                    eval_domains=["illumination", "viewpoint"],
                    epochs=config.get("deep_epochs", 10),
                    lr=config.get("lr", 1e-4),
                    batch_size=config.get("batch_size", 32),
                    device=device,
                    logger=logger,
                )
                all_results["deep"][f"{model_name}_{train_domain}"] = results

    # 3. Continual learning comparison
    if config.get("eval_continual", True):
        all_results["continual"] = {}

        for source, target in [("illumination", "viewpoint"), ("viewpoint", "illumination")]:
            results = evaluate_continual_learning(
                source_domain=source,
                target_domain=target,
                methods=config.get("continual_methods", ["naive", "ewc", "lwf", "si"]),
                epochs_source=config.get("epochs_source", 10),
                epochs_target=config.get("epochs_target", 10),
                lr=config.get("lr", 1e-4),
                batch_size=config.get("batch_size", 32),
                device=device,
                logger=logger,
            )
            all_results["continual"][f"{source}_to_{target}"] = results

    # 4. MAML evaluation
    if config.get("eval_maml", True):
        all_results["maml"] = {}

        for source, target in [("illumination", "viewpoint"), ("viewpoint", "illumination")]:
            results = evaluate_maml(
                source_domain=source,
                target_domain=target,
                epochs_pretrain=config.get("epochs_source", 10),
                epochs_meta=config.get("meta_epochs", 5),
                lr_inner=config.get("meta_lr_inner", 1e-3),
                lr_outer=config.get("meta_lr_outer", 1e-4),
                batch_size=config.get("batch_size", 32),
                device=device,
                logger=logger,
            )
            all_results["maml"][f"{source}_to_{target}"] = results

    # Save results
    results_path = logger.log_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {results_path}")

    # Generate summary plots
    print("\nGenerating summary plots...")
    _generate_plots(all_results, logger)

    logger.finish()

    return all_results


def _generate_plots(results: Dict, logger: Logger):
    """Generate summary visualization plots."""

    # Traditional matcher comparison
    if "traditional" in results:
        trad_mma = {}
        for name, domains in results["traditional"].items():
            trad_mma[name] = {
                "mma_1px": np.mean([d.get("mma_1px", 0) for d in domains.values()]),
                "mma_3px": np.mean([d.get("mma_3px", 0) for d in domains.values()]),
                "mma_5px": np.mean([d.get("mma_5px", 0) for d in domains.values()]),
                "mma_10px": np.mean([d.get("mma_10px", 0) for d in domains.values()]),
            }

        fig = plot_mma_comparison(trad_mma, title="Traditional Matchers: MMA Comparison")
        logger.log_figure("traditional_mma", fig, 0)

    # Forgetting comparison
    if "continual" in results:
        for transfer, methods in results["continual"].items():
            forgetting_data = {m: d["forgetting_curve"]["source_acc"] for m, d in methods.items() if "forgetting_curve" in d}
            if forgetting_data:
                epochs = methods[list(methods.keys())[0]]["forgetting_curve"]["epochs"]
                fig = plot_forgetting_curve(
                    forgetting_data, epochs,
                    title=f"Forgetting Curve: {transfer.replace('_', ' ')}"
                )
                logger.log_figure(f"forgetting_{transfer}", fig, 0)


def main():
    parser = argparse.ArgumentParser(description="Run feature matching benchmark")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark (fewer epochs)")
    args = parser.parse_args()

    # Default config
    config = {
        "project": "meta-feature-matching",
        "run_name": "benchmark",
        "use_wandb": True,
        "eval_traditional": True,
        "eval_deep": True,
        "eval_continual": True,
        "eval_maml": True,
        "deep_models": ["resnet50", "lightweight"],
        "continual_methods": ["naive", "ewc", "lwf", "si"],
        "deep_epochs": 10,
        "epochs_source": 10,
        "epochs_target": 10,
        "meta_epochs": 5,
        "lr": 1e-4,
        "meta_lr_inner": 1e-3,
        "meta_lr_outer": 1e-4,
        "batch_size": 32,
    }

    # Load config from file
    if args.config:
        with open(args.config) as f:
            config.update(yaml.safe_load(f))

    if args.quick:
        config.update({
            "deep_epochs": 2,
            "epochs_source": 2,
            "epochs_target": 2,
            "meta_epochs": 2,
            "deep_models": ["resnet50"],
            "continual_methods": ["naive", "ewc"],
            "batch_size": 64,       # ADD: Larger batch
            "max_samples": 3000,    # ADD: Limit samples
            "num_workers": 4,       # ADD: Parallel loading
        })

    run_full_benchmark(config)


if __name__ == "__main__":
    main()

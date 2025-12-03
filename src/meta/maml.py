"""
Model-Agnostic Meta-Learning (MAML) for Feature Descriptors

Implements first-order MAML for fast adaptation to new domains.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional, Dict
from copy import deepcopy
import higher


class MAMLTrainer:
    """
    MAML trainer for descriptor learning.
    
    Args:
        model: The descriptor network
        lr_inner: Learning rate for inner loop adaptation
        lr_outer: Learning rate for outer loop (meta) optimization
        num_inner_steps: Number of gradient steps in inner loop
        first_order: Use first-order approximation (faster, often works well)
        device: torch device
    """

    def __init__(
        self,
        model: nn.Module,
        lr_inner: float = 1e-3,
        lr_outer: float = 1e-4,
        num_inner_steps: int = 3,
        first_order: bool = False,
        device: str = "cuda",
    ):
        self.model = model
        self.lr_inner = lr_inner
        self.num_inner_steps = num_inner_steps
        self.first_order = first_order
        self.device = device

        self.outer_optimizer = torch.optim.Adam(model.parameters(), lr=lr_outer)

    def inner_loop(
        self,
        fmodel: nn.Module,
        diffopt,
        support: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        loss_fn: nn.Module,
    ) -> float:
        """
        Perform inner loop adaptation on support set.
        """
        total_loss = 0

        for step in range(self.num_inner_steps):
            for a, p, n in support:
                a = a.to(self.device)
                p = p.to(self.device)
                n = n.to(self.device)

                da, dp, dn = fmodel(a), fmodel(p), fmodel(n)
                loss = loss_fn(da, dp, dn)
                diffopt.step(loss)
                total_loss += loss.item()

        return total_loss / (self.num_inner_steps * len(support))

    def meta_train_step(
        self,
        tasks: List[Tuple[List, List]],
        loss_fn: nn.Module,
    ) -> Dict[str, float]:
        """
        Perform one meta-training step.
        
        Args:
            tasks: List of (support, query) tuples
            loss_fn: Loss function (e.g., TripletMarginLoss)
            
        Returns:
            Dict with loss values
        """
        meta_loss = 0.0
        inner_losses = []

        for support, query in tasks:
            # Create functional model for inner loop
            inner_opt = torch.optim.SGD(self.model.parameters(), lr=self.lr_inner)

            with higher.innerloop_ctx(
                self.model,
                inner_opt,
                track_higher_grads=not self.first_order,
            ) as (fmodel, diffopt):
                
                # Inner loop: adapt on support set
                inner_loss = self.inner_loop(fmodel, diffopt, support, loss_fn)
                inner_losses.append(inner_loss)

                # Compute loss on query set
                query_loss = 0
                for a, p, n in query:
                    a = a.to(self.device)
                    p = p.to(self.device)
                    n = n.to(self.device)

                    da, dp, dn = fmodel(a), fmodel(p), fmodel(n)
                    query_loss += loss_fn(da, dp, dn)

                meta_loss += query_loss / len(query)

        # Average over tasks
        meta_loss /= len(tasks)

        # Outer loop update
        self.outer_optimizer.zero_grad()
        meta_loss.backward()
        self.outer_optimizer.step()

        return {
            "meta_loss": meta_loss.item(),
            "inner_loss": sum(inner_losses) / len(inner_losses),
        }

    def adapt(
        self,
        support_loader: DataLoader,
        loss_fn: nn.Module,
        num_steps: Optional[int] = None,
    ) -> nn.Module:
        """
        Adapt model to new domain using support data.
        
        Returns a new adapted model (does not modify original).
        
        Args:
            support_loader: DataLoader for adaptation data
            loss_fn: Loss function
            num_steps: Number of adaptation steps (default: self.num_inner_steps)
        """
        num_steps = num_steps or self.num_inner_steps

        # Clone model
        adapted_model = deepcopy(self.model)
        adapted_model.train()

        optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.lr_inner)

        step = 0
        while step < num_steps:
            for a, p, n in support_loader:
                if step >= num_steps:
                    break

                a = a.to(self.device)
                p = p.to(self.device)
                n = n.to(self.device)

                da = adapted_model(a)
                dp = adapted_model(p)
                dn = adapted_model(n)

                loss = loss_fn(da, dp, dn)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                step += 1

        adapted_model.eval()
        return adapted_model


class FOMAMLTrainer(MAMLTrainer):
    """First-Order MAML (faster, often comparable performance)."""

    def __init__(self, *args, **kwargs):
        kwargs["first_order"] = True
        super().__init__(*args, **kwargs)


def create_meta_batch(
    loaders: Dict[str, DataLoader],
    support_batches: int = 2,
    query_batches: int = 2,
) -> List[Tuple[List, List]]:
    """
    Create a meta-batch from multiple domain loaders.
    
    Args:
        loaders: Dict mapping domain name to DataLoader
        support_batches: Number of batches for support set per task
        query_batches: Number of batches for query set per task
        
    Returns:
        List of (support, query) tuples
    """
    tasks = []

    for domain, loader in loaders.items():
        iterator = iter(loader)

        support = []
        query = []

        try:
            for _ in range(support_batches):
                support.append(next(iterator))
            for _ in range(query_batches):
                query.append(next(iterator))

            tasks.append((support, query))

        except StopIteration:
            # Not enough data, skip this domain
            continue

    return tasks


def meta_adaptation_curve(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    lr_inner: float,
    device: str,
    steps: List[int] = [0, 1, 2, 5, 10],
    eval_fn=None,
) -> Dict[str, List[float]]:
    """
    Evaluate how quickly model adapts to new domain.
    
    Args:
        model: The meta-learned model
        loader: DataLoader for target domain
        loss_fn: Loss function
        lr_inner: Inner loop learning rate
        device: torch device
        steps: List of step counts to evaluate at
        eval_fn: Optional evaluation function (default: triplet accuracy)
        
    Returns:
        Dict with steps and corresponding accuracies
    """
    from src.evaluation.metrics import triplet_accuracy

    if eval_fn is None:
        eval_fn = lambda m: triplet_accuracy(m, loader, device).accuracy

    # Save initial state
    init_state = {k: v.clone() for k, v in model.state_dict().items()}

    results = {"steps": steps, "accuracies": []}

    for s in steps:
        # Restore model
        model.load_state_dict({k: v.clone() for k, v in init_state.items()})

        # Adapt for s steps
        if s > 0:
            model.train()
            optimizer = torch.optim.SGD(model.parameters(), lr=lr_inner)

            step = 0
            while step < s:
                for a, p, n in loader:
                    if step >= s:
                        break

                    a, p, n = a.to(device), p.to(device), n.to(device)
                    da, dp, dn = model(a), model(p), model(n)
                    loss = loss_fn(da, dp, dn)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    step += 1

        # Evaluate
        model.eval()
        acc = eval_fn(model)
        results["accuracies"].append(acc)

    # Restore initial state
    model.load_state_dict(init_state)

    return results
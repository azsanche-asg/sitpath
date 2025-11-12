from __future__ import annotations

from typing import Callable, Iterable, Optional

import torch
from tqdm.auto import tqdm


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: Iterable,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    device: torch.device,
    metrics_fn: Callable,
    log_fn: Optional[Callable] = None,
):
    """Run a single training epoch with progress feedback."""

    model.train()
    total_loss = 0.0
    progress = tqdm(dataloader, desc="train", leave=False)
    for batch in progress:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / max(len(dataloader), 1)
    metrics = metrics_fn(model, dataloader, device=device)
    log_payload = {"phase": "train", "loss": avg_loss, **metrics}
    if log_fn:
        log_fn(log_payload)
    else:
        print(log_payload)
    return avg_loss, metrics


def evaluate(
    model: torch.nn.Module,
    dataloader: Iterable,
    metrics_fn: Callable,
    device: torch.device,
    log_fn: Optional[Callable] = None,
):
    """Evaluate the model on a dataloader."""

    model.eval()
    total_loss = 0.0
    progress = tqdm(dataloader, desc="eval", leave=False)
    with torch.no_grad():
        for batch in progress:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = torch.nn.functional.mse_loss(outputs, targets)
            total_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / max(len(dataloader), 1)
    metrics = metrics_fn(model, dataloader, device=device)
    log_payload = {"phase": "eval", "loss": avg_loss, **metrics}
    if log_fn:
        log_fn(log_payload)
    else:
        print(log_payload)
    return avg_loss, metrics

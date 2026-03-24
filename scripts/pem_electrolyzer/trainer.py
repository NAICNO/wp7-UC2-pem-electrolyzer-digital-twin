"""
Training Module for Teacher Model.

Implements training loop for the HybridPhysicsMLP teacher model with:
- SGD optimizer with momentum (KEY FOR BEST OOD GENERALIZATION!)
- CosineAnnealingLR scheduler for stable training
- Early stopping based on validation MAE
- Progress tracking and checkpointing
"""

import copy
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Tuple, Optional
import time

try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm


def train_teacher(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    stats: Dict,
    epochs: int = 100,
    lr: float = 0.01,
    patience: int = 30,
    save_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict]:
    """
    Train the teacher model (HybridPhysicsMLP) from scratch.

    Args:
        model: HybridPhysicsMLP model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        stats: Normalization statistics from data loader
        epochs: Maximum training epochs
        lr: Initial learning rate
        patience: Early stopping patience
        save_dir: Directory to save checkpoints
        verbose: Print training progress

    Returns:
        model: Trained model (best checkpoint)
        history: Training history dictionary
    """
    device = next(model.parameters()).device

    if verbose:
        print("\n" + "="*60)
        print("Training Teacher Model (HybridPhysicsMLP)")
        print("="*60)
        print(f"Device: {device}")
        print(f"Parameters: {model.count_parameters():,}")
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {lr}")
        print(f"Early stopping patience: {patience}")

    # Set normalization statistics (using average pressure, matching original model)
    model.set_normalization_stats(
        i_mean=stats['i_mean'], i_std=stats['i_std'],
        P_mean=stats['P_mean'], P_std=stats['P_std'],
        T_mean=stats['T_mean'], T_std=stats['T_std'],
        V_mean=stats['V_mean'], V_std=stats['V_std'],
    )

    # Setup optimizer and scheduler (SGD + CosineAnnealingLR is KEY for OOD!)
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0001)
    criterion = nn.MSELoss()

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae_mV': [],
        'lr': [],
        'best_epoch': 0,
        'best_val_mae_mV': float('inf'),
    }

    # Early stopping
    best_val_mae = float('inf')
    best_state = None
    patience_counter = 0

    # Create save directory
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    pbar = tqdm(range(epochs), desc="Teacher", disable=not verbose,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}")

    for epoch in pbar:
        # ================================================================
        # Training Phase
        # ================================================================
        model.train()
        train_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            # Forward pass
            V_pred, _ = model(X_batch)
            loss = criterion(V_pred, y_batch)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= n_batches

        # ================================================================
        # Validation Phase
        # ================================================================
        model.eval()
        val_loss = 0.0
        val_errors = []
        n_val_batches = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                V_pred, _ = model(X_batch)
                loss = criterion(V_pred, y_batch)
                val_loss += loss.item()

                # MAE in mV
                errors = torch.abs(V_pred - y_batch) * 1000
                val_errors.extend(errors.cpu().numpy())
                n_val_batches += 1

        val_loss /= n_val_batches
        val_mae_mV = sum(val_errors) / len(val_errors)

        # Update scheduler (CosineAnnealingLR steps by epoch, not metric)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae_mV'].append(val_mae_mV)
        history['lr'].append(current_lr)

        # ================================================================
        # Early Stopping Check
        # ================================================================
        if val_mae_mV < best_val_mae:
            best_val_mae = val_mae_mV
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            history['best_epoch'] = epoch
            history['best_val_mae_mV'] = val_mae_mV

            # Save checkpoint
            if save_dir:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_mae_mV': val_mae_mV,
                    'stats': stats,
                }, save_dir / 'best_teacher.pt')
        else:
            patience_counter += 1

        # Progress output via tqdm postfix
        star = " *" if patience_counter == 0 else ""
        pbar.set_postfix_str(
            f"loss={train_loss:.5f} val={val_mae_mV:.1f}mV best={best_val_mae:.1f}mV lr={current_lr:.5f}{star}"
        )

        # Early stopping
        if patience_counter >= patience:
            pbar.set_postfix_str(f"Early stop @ {epoch+1} | best={best_val_mae:.1f}mV")
            break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Training summary
    total_time = time.time() - start_time
    history['total_time'] = total_time

    if verbose:
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"Best epoch: {history['best_epoch']+1}")
        print(f"Best val MAE: {history['best_val_mae_mV']:.2f} mV")
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

    return model, history


if __name__ == "__main__":
    """Quick test of trainer."""
    print("Trainer module loaded successfully!")
    print("Use train_teacher() to train the model.")

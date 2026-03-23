"""
Knowledge Distillation Module for Student Model.

Implements knowledge distillation from teacher (HybridPhysicsMLP) to
student (PhysicsHybrid12Param) using soft label loss:

Loss = α * MSE(student, labels) + (1-α) * MSE(student, teacher)

Key insight from experiments:
- α = 0.1 gives best OOD generalization (more weight on teacher knowledge)
- Lower α means student learns more from teacher's implicit physics
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Tuple, Optional
import time

try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm


def train_student_distillation(
    student: nn.Module,
    teacher: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    alpha: float = 0.1,
    epochs: int = 200,
    lr: float = 0.001,
    patience: int = 40,
    save_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict]:
    """
    Train student model via knowledge distillation from teacher.

    Loss = α * MSE(student, labels) + (1-α) * MSE(student, teacher)

    Key insight: Lower α (more teacher weight) gives better OOD generalization
    because the teacher's implicit physics knowledge transfers to the student.

    Args:
        student: PhysicsHybrid12Param student model
        teacher: Trained HybridPhysicsMLP teacher model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        alpha: Balance between label loss and distillation loss
               α=0.1 means 10% labels, 90% teacher (best for OOD)
        epochs: Maximum training epochs
        lr: Initial learning rate
        patience: Early stopping patience
        save_dir: Directory to save checkpoints
        verbose: Print training progress

    Returns:
        student: Trained student model (best checkpoint)
        history: Training history dictionary
    """
    device = next(teacher.parameters()).device
    student = student.to(device)
    teacher.eval()  # Teacher in eval mode (frozen)

    if verbose:
        print("\n" + "="*60)
        print("Training Student via Knowledge Distillation")
        print("="*60)
        print(f"Device: {device}")
        print(f"Student parameters: {student.count_parameters()}")
        print(f"Alpha (label weight): {alpha}")
        print(f"  -> {alpha*100:.0f}% labels, {(1-alpha)*100:.0f}% teacher")
        print(f"Epochs: {epochs}")
        print(f"Early stopping patience: {patience}")

    # Setup optimizer and scheduler
    optimizer = Adam(student.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()

    # Training history
    history = {
        'train_loss': [],
        'train_label_loss': [],
        'train_distill_loss': [],
        'val_loss': [],
        'val_mae_mV': [],
        'lr': [],
        'alpha': alpha,
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

    pbar = tqdm(range(epochs), desc="Student", disable=not verbose,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}")

    for epoch in pbar:
        # ================================================================
        # Training Phase with Distillation
        # ================================================================
        student.train()
        train_loss = 0.0
        label_loss_total = 0.0
        distill_loss_total = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            # Student prediction
            V_student = student(X_batch)

            # Teacher prediction (no gradient)
            with torch.no_grad():
                V_teacher, _ = teacher(X_batch)

            # Distillation loss
            # Loss = α * MSE(student, labels) + (1-α) * MSE(student, teacher)
            label_loss = criterion(V_student, y_batch)
            distill_loss = criterion(V_student, V_teacher)
            loss = alpha * label_loss + (1 - alpha) * distill_loss

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            label_loss_total += label_loss.item()
            distill_loss_total += distill_loss.item()
            n_batches += 1

        train_loss /= n_batches
        label_loss_total /= n_batches
        distill_loss_total /= n_batches

        # ================================================================
        # Validation Phase
        # ================================================================
        student.eval()
        val_loss = 0.0
        val_errors = []
        n_val_batches = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                V_pred = student(X_batch)
                loss = criterion(V_pred, y_batch)
                val_loss += loss.item()

                # MAE in mV
                errors = torch.abs(V_pred - y_batch) * 1000
                val_errors.extend(errors.cpu().numpy())
                n_val_batches += 1

        val_loss /= n_val_batches
        val_mae_mV = sum(val_errors) / len(val_errors)

        # Update scheduler
        scheduler.step(val_mae_mV)
        current_lr = optimizer.param_groups[0]['lr']

        # Record history
        history['train_loss'].append(train_loss)
        history['train_label_loss'].append(label_loss_total)
        history['train_distill_loss'].append(distill_loss_total)
        history['val_loss'].append(val_loss)
        history['val_mae_mV'].append(val_mae_mV)
        history['lr'].append(current_lr)

        # ================================================================
        # Early Stopping Check
        # ================================================================
        if val_mae_mV < best_val_mae:
            best_val_mae = val_mae_mV
            # Deep copy state_dict - .copy() only does shallow copy, tensors are shared!
            best_state = {k: v.clone() for k, v in student.state_dict().items()}
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
                    'alpha': alpha,
                    'physics_params': student.get_physics_params(),
                    'hybrid_params': student.get_hybrid_params(),
                }, save_dir / f'best_student_alpha{alpha}.pt')
        else:
            patience_counter += 1

        # Progress output via tqdm postfix
        star = " *" if patience_counter == 0 else ""
        pbar.set_postfix_str(
            f"loss={train_loss:.5f} L={label_loss_total:.4f} D={distill_loss_total:.4f} "
            f"val={val_mae_mV:.1f}mV best={best_val_mae:.1f}mV lr={current_lr:.5f}{star}"
        )

        # Early stopping
        if patience_counter >= patience:
            pbar.set_postfix_str(f"Early stop @ {epoch+1} | best={best_val_mae:.1f}mV")
            break

    # Load best model
    if best_state is not None:
        student.load_state_dict(best_state)

    # Training summary
    total_time = time.time() - start_time
    history['total_time'] = total_time

    if verbose:
        print("\n" + "="*60)
        print("Distillation Complete!")
        print("="*60)
        print(f"Best epoch: {history['best_epoch']+1}")
        print(f"Best val MAE: {history['best_val_mae_mV']:.2f} mV")
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

        # Print learned parameters
        print("\nLearned Physics Parameters:")
        for k, v in student.get_physics_params().items():
            print(f"  {k}: {v:.6f}")

        print("\nHybrid Correction Parameters:")
        for k, v in student.get_hybrid_params().items():
            print(f"  {k}: {v:.6f}")

    return student, history


if __name__ == "__main__":
    """Quick test of distillation module."""
    print("Distillation module loaded successfully!")
    print("Use train_student_distillation() to train the student model.")

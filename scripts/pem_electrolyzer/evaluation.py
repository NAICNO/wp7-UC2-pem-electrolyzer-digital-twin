"""
Evaluation Module for PEM Electrolyzer Models.

Provides functions to evaluate models on different datasets:
- evaluate_model: Evaluate on a single DataLoader
- evaluate_ood: Evaluate on Test2 and Test3 OOD datasets
- compare_models: Compare teacher vs student

Metrics:
- MAE (Mean Absolute Error) in millivolts [mV]
- OOD Average: (Test2_MAE + Test3_MAE) / 2
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple

from dataloader import load_ood_minimal


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = None,
    verbose: bool = False,
) -> float:
    """
    Evaluate model on a dataset.

    Args:
        model: Model to evaluate (teacher or student)
        dataloader: DataLoader with (X, y) batches
        device: Device to use (auto-detect if None)
        verbose: Print evaluation details

    Returns:
        mae_mV: Mean Absolute Error in millivolts
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    errors = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Handle both teacher (returns tuple) and student (returns tensor)
            output = model(X_batch)
            if isinstance(output, tuple):
                V_pred = output[0]  # Teacher returns (V, params)
            else:
                V_pred = output  # Student returns V directly

            # MAE in mV
            batch_errors = torch.abs(V_pred - y_batch) * 1000
            errors.extend(batch_errors.cpu().numpy())

    mae_mV = sum(errors) / len(errors)

    if verbose:
        print(f"  Samples: {len(errors):,}")
        print(f"  MAE: {mae_mV:.2f} mV")

    return mae_mV


def evaluate_ood(
    model: nn.Module,
    device: str = None,
    batch_size: int = 4096,
    verbose: bool = True,
    data_dir: str = 'dataset/',
) -> Dict[str, float]:
    """
    Evaluate model on Out-of-Distribution datasets (Test2 and Test3).

    Uses MINIMAL filtering to preserve OOD characteristics:
    - Only removes NaN, negative pressures, cold startup
    - Broader voltage range (1.0-2.5V)

    Args:
        model: Model to evaluate
        device: Device to use
        batch_size: Batch size for DataLoader
        verbose: Print evaluation results
        data_dir: Directory containing data files

    Returns:
        results: Dictionary with Test2, Test3, and OOD Average MAE
    """
    if device is None:
        device = next(model.parameters()).device

    if verbose:
        print("\n" + "="*60)
        print("Evaluating on OOD Datasets (MINIMAL filtering)")
        print("="*60)

    results = {}

    # Evaluate Test2
    if verbose:
        print("\n[Test2 Evaluation]")
    test2_loader, test2_info = load_ood_minimal(
        "test2", data_dir=data_dir, device=device, batch_size=batch_size, verbose=verbose
    )
    test2_mae = evaluate_model(model, test2_loader, device, verbose=verbose)
    results['test2_mae_mV'] = test2_mae
    results['test2_samples'] = test2_info['n_samples']

    # Evaluate Test3
    if verbose:
        print("\n[Test3 Evaluation]")
    test3_loader, test3_info = load_ood_minimal(
        "test3", data_dir=data_dir, device=device, batch_size=batch_size, verbose=verbose
    )
    test3_mae = evaluate_model(model, test3_loader, device, verbose=verbose)
    results['test3_mae_mV'] = test3_mae
    results['test3_samples'] = test3_info['n_samples']

    # OOD Average
    ood_avg = (test2_mae + test3_mae) / 2
    results['ood_avg_mV'] = ood_avg

    if verbose:
        print("\n" + "="*60)
        print("OOD Evaluation Summary")
        print("="*60)
        print(f"  Test2 MAE: {test2_mae:.2f} mV ({test2_info['n_samples']:,} samples)")
        print(f"  Test3 MAE: {test3_mae:.2f} mV ({test3_info['n_samples']:,} samples)")
        print(f"  ─────────────────────────────")
        print(f"  OOD Average: {ood_avg:.2f} mV")

    return results


def compare_models(
    teacher: nn.Module,
    student: nn.Module,
    val_loader: DataLoader,
    device: str = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Compare teacher and student models on validation and OOD datasets.

    Args:
        teacher: Trained teacher model
        student: Trained student model
        val_loader: Validation DataLoader
        device: Device to use
        verbose: Print comparison results

    Returns:
        comparison: Dictionary with results for both models
    """
    if device is None:
        device = next(teacher.parameters()).device

    if verbose:
        print("\n" + "="*60)
        print("Model Comparison: Teacher vs Student")
        print("="*60)

    comparison = {
        'teacher': {},
        'student': {},
    }

    # Validation
    if verbose:
        print("\n[Validation]")
    teacher_val_mae = evaluate_model(teacher, val_loader, device)
    student_val_mae = evaluate_model(student, val_loader, device)
    comparison['teacher']['val_mae_mV'] = teacher_val_mae
    comparison['student']['val_mae_mV'] = student_val_mae

    if verbose:
        print(f"  Teacher: {teacher_val_mae:.2f} mV")
        print(f"  Student: {student_val_mae:.2f} mV")

    # OOD
    if verbose:
        print("\n[OOD Evaluation]")

    # Teacher OOD
    if verbose:
        print("\nTeacher:")
    teacher_ood = evaluate_ood(teacher, device, verbose=False)
    comparison['teacher'].update(teacher_ood)

    # Student OOD
    if verbose:
        print("Student:")
    student_ood = evaluate_ood(student, device, verbose=False)
    comparison['student'].update(student_ood)

    if verbose:
        print("\n" + "="*60)
        print("Comparison Summary")
        print("="*60)
        print(f"{'Metric':<20} {'Teacher':>12} {'Student':>12} {'Diff':>12}")
        print("-"*56)
        print(f"{'Val MAE (mV)':<20} {teacher_val_mae:>12.2f} {student_val_mae:>12.2f} {student_val_mae - teacher_val_mae:>+12.2f}")
        print(f"{'Test2 MAE (mV)':<20} {teacher_ood['test2_mae_mV']:>12.2f} {student_ood['test2_mae_mV']:>12.2f} {student_ood['test2_mae_mV'] - teacher_ood['test2_mae_mV']:>+12.2f}")
        print(f"{'Test3 MAE (mV)':<20} {teacher_ood['test3_mae_mV']:>12.2f} {student_ood['test3_mae_mV']:>12.2f} {student_ood['test3_mae_mV'] - teacher_ood['test3_mae_mV']:>+12.2f}")
        print(f"{'OOD Avg (mV)':<20} {teacher_ood['ood_avg_mV']:>12.2f} {student_ood['ood_avg_mV']:>12.2f} {student_ood['ood_avg_mV'] - teacher_ood['ood_avg_mV']:>+12.2f}")

        # Highlight improvement
        if student_ood['ood_avg_mV'] < teacher_ood['ood_avg_mV']:
            improvement = (teacher_ood['ood_avg_mV'] - student_ood['ood_avg_mV']) / teacher_ood['ood_avg_mV'] * 100
            print(f"\n>>> Student improves OOD by {improvement:.1f}%!")

    return comparison


if __name__ == "__main__":
    """Quick test of evaluation module."""
    print("Evaluation module loaded successfully!")
    print("\nAvailable functions:")
    print("  - evaluate_model(model, dataloader): Returns MAE in mV")
    print("  - evaluate_ood(model): Evaluates on Test2 and Test3")
    print("  - compare_models(teacher, student, val_loader): Full comparison")

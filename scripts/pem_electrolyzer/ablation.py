#!/usr/bin/env python3
"""
Ablation Study for PEM Electrolyzer PINN.

7 experiments x 3 seeds = 21 training runs:
1. Pure MLP (no physics)
2. Big MLP (50K params, no physics)
3. Transformer (50K params, no physics)
4. Teacher (physics + MLP)
5. Pure Physics (student with alpha=1.0, no distillation)
6. Student + Distillation (alpha=0.1) + Keep-out validation
7. Student + Distillation (alpha=0.1) + Random validation
"""

import json
import os
import sys
import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# Add script directory to path for bare imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import get_model, HybridPhysicsMLP, PhysicsHybrid12Param
from dataloader import load_test4_training
from trainer import train_teacher
from distillation import train_student_distillation
from evaluation import evaluate_ood, evaluate_model


SEEDS = [42, 91, 142]
EXPERIMENTS = [
    'pure_mlp', 'big_mlp', 'transformer',
    'teacher', 'pure_physics',
    'student_keepout', 'student_random',
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_baseline(model_name, train_loader, val_loader, stats, device, epochs=100, lr=0.01):
    """Train a baseline model (PureMLP, BigMLP, Transformer)."""
    model = get_model(model_name, device=device).to(device)

    # Set normalization stats
    if hasattr(model, 'set_normalization_stats'):
        model.set_normalization_stats(
            i_mean=stats['i_mean'], i_std=stats['i_std'],
            P_mean=stats['P_mean'], P_std=stats['P_std'],
            T_mean=stats['T_mean'], T_std=stats['T_std'],
        )

    # Train with SGD (same as teacher for fair comparison)
    from torch.optim import SGD
    from torch.optim.lr_scheduler import CosineAnnealingLR
    import copy
    import time

    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0001)
    criterion = torch.nn.MSELoss()

    best_val_mae = float('inf')
    best_state = None
    patience_counter = 0
    start = time.time()

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            V_pred = model(X_batch)
            loss = criterion(V_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        # Validate
        model.eval()
        val_errors = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                V_pred = model(X_batch)
                errors = torch.abs(V_pred - y_batch) * 1000
                val_errors.extend(errors.cpu().numpy())

        val_mae = sum(val_errors) / len(val_errors)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 30:
            break

    if best_state:
        model.load_state_dict(best_state)

    return model, {'best_val_mae_mV': best_val_mae, 'total_time': time.time() - start}


def run_single_experiment(exp_name, seed, data_dir, device, epochs=100):
    """Run a single experiment + seed combination."""
    set_seed(seed)

    use_keepout = 'random' not in exp_name
    train_loader, val_loader, stats = load_test4_training(
        data_dir=data_dir, device=device, batch_size=4096,
        verbose=False, seed=seed, use_keepout=use_keepout
    )

    if exp_name in ['pure_mlp', 'big_mlp', 'transformer']:
        model, history = train_baseline(exp_name, train_loader, val_loader, stats, device, epochs)
    elif exp_name == 'teacher':
        model = HybridPhysicsMLP(device=device)
        model, history = train_teacher(model, train_loader, val_loader, stats,
                                        epochs=epochs, lr=0.01, patience=30, verbose=False)
    elif exp_name == 'pure_physics':
        teacher = HybridPhysicsMLP(device=device)
        teacher, _ = train_teacher(teacher, train_loader, val_loader, stats,
                                    epochs=epochs, lr=0.01, patience=30, verbose=False)
        model = PhysicsHybrid12Param().to(device)
        model, history = train_student_distillation(model, teacher, train_loader, val_loader,
                                                     alpha=1.0, epochs=200, lr=0.001,
                                                     patience=40, verbose=False)
    elif exp_name in ['student_keepout', 'student_random']:
        teacher = HybridPhysicsMLP(device=device)
        teacher, _ = train_teacher(teacher, train_loader, val_loader, stats,
                                    epochs=epochs, lr=0.01, patience=30, verbose=False)
        model = PhysicsHybrid12Param().to(device)
        model, history = train_student_distillation(model, teacher, train_loader, val_loader,
                                                     alpha=0.1, epochs=200, lr=0.001,
                                                     patience=40, verbose=False)
    else:
        raise ValueError(f"Unknown experiment: {exp_name}")

    # Evaluate OOD
    ood = evaluate_ood(model, device=device, verbose=False, data_dir=data_dir)

    result = {
        'experiment': exp_name,
        'seed': seed,
        'val_mae_mV': history['best_val_mae_mV'],
        'test2_mae_mV': ood['test2_mae_mV'],
        'test3_mae_mV': ood['test3_mae_mV'],
        'ood_avg_mV': ood['ood_avg_mV'],
        'training_time_s': history['total_time'],
    }

    return result


def run_ablation(args, device, output_dir):
    """Run full ablation study."""
    print("\n" + "="*70)
    print("ABLATION STUDY: 7 Experiments x 3 Seeds = 21 Runs")
    print("="*70)

    ablation_dir = output_dir / 'ablation'
    ablation_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for exp_name in EXPERIMENTS:
        print(f"\n--- {exp_name} ---")
        for seed in SEEDS:
            print(f"  Seed {seed}...", end=" ", flush=True)
            result = run_single_experiment(exp_name, seed, args.data_dir, device, args.epochs)
            all_results.append(result)

            # Save per-experiment result
            result_file = ablation_dir / f"{exp_name}_seed{seed}.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)

            print(f"OOD={result['ood_avg_mV']:.1f} mV")

    # Print summary table
    print("\n" + "="*70)
    print("ABLATION RESULTS")
    print("="*70)
    print(f"{'Experiment':<20} {'Seed':>6} {'Val MAE':>10} {'Test2':>10} {'Test3':>10} {'OOD Avg':>10}")
    print("-"*70)
    for r in all_results:
        print(f"{r['experiment']:<20} {r['seed']:>6} {r['val_mae_mV']:>10.2f} {r['test2_mae_mV']:>10.2f} {r['test3_mae_mV']:>10.2f} {r['ood_avg_mV']:>10.2f}")

    # Save all results
    summary_file = ablation_dir / 'ablation_results.json'
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {summary_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PEM Electrolyzer PINN Ablation Study")
    parser.add_argument("--data-dir", default="dataset/")
    parser.add_argument("--output-dir", default="results/")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(args.output_dir)
    run_ablation(args, device, output_dir)

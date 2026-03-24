#!/usr/bin/env python3
"""
PEM Electrolyzer PINN - MLOPS Demonstrator

Main entry point for training and evaluating physics-informed neural networks
for PEM electrolyzer voltage prediction.

Modes:
  full       - Train teacher, distill student, evaluate OOD (default)
  quick-test - 5 epochs, fast verification
  teacher-only - Train only teacher model
  ablation   - Run ablation study (7 experiments x 3 seeds)

Usage:
  python scripts/pem_electrolyzer/main.py --mode full --epochs 100 --seed 42
  python scripts/pem_electrolyzer/main.py --mode quick-test
  python scripts/pem_electrolyzer/main.py --mode ablation --device cuda
"""

import argparse
import json
import os
import random
import sys
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# Add script directory to path for bare imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import HybridPhysicsMLP, PhysicsHybrid12Param, get_model
from dataloader import load_test4_training, load_ood_minimal
from trainer import train_teacher
from distillation import train_student_distillation
from evaluation import evaluate_ood, compare_models


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def run_full(args, device, output_dir):
    """Full pipeline: train teacher, distill student, evaluate OOD."""
    # 1. Load data
    print("\n[Step 1/6] Loading Training Data...")
    train_loader, val_loader, stats = load_test4_training(
        data_dir=args.data_dir, device=device, batch_size=args.batch_size,
        verbose=True, seed=args.seed
    )

    # 2. Train teacher
    print("\n[Step 2/6] Training Teacher (HybridPhysicsMLP)...")
    teacher = HybridPhysicsMLP(device=device)
    teacher, teacher_history = train_teacher(
        model=teacher, train_loader=train_loader, val_loader=val_loader,
        stats=stats, epochs=args.epochs, lr=args.lr, patience=30,
        save_dir=output_dir, verbose=True
    )

    # 3. Evaluate teacher OOD
    print("\n[Step 3/6] Evaluating Teacher on OOD...")
    teacher_ood = evaluate_ood(teacher, device=device, verbose=True, data_dir=args.data_dir)

    # 4. Distill student
    print(f"\n[Step 4/6] Training Student (alpha={args.alpha})...")
    student = PhysicsHybrid12Param().to(device)
    student, student_history = train_student_distillation(
        student=student, teacher=teacher, train_loader=train_loader,
        val_loader=val_loader, alpha=args.alpha, epochs=args.epochs,
        lr=args.lr, patience=40, save_dir=output_dir, verbose=True
    )

    # 5. Evaluate student OOD
    print("\n[Step 5/6] Evaluating Student on OOD...")
    student_ood = evaluate_ood(student, device=device, verbose=True, data_dir=args.data_dir)

    # 6. Compare and save
    print("\n[Step 6/6] Final Comparison...")
    comparison = compare_models(teacher, student, val_loader, device=device, verbose=True)

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'seed': args.seed, 'epochs': args.epochs, 'batch_size': args.batch_size,
            'lr': args.lr, 'alpha': args.alpha, 'device': device, 'mode': 'full',
        },
        'teacher': {
            'val_mae_mV': teacher_history['best_val_mae_mV'],
            'best_epoch': teacher_history['best_epoch'],
            'training_time_s': teacher_history['total_time'],
            'test2_mae_mV': teacher_ood['test2_mae_mV'],
            'test3_mae_mV': teacher_ood['test3_mae_mV'],
            'ood_avg_mV': teacher_ood['ood_avg_mV'],
        },
        'student': {
            'val_mae_mV': student_history['best_val_mae_mV'],
            'best_epoch': student_history['best_epoch'],
            'training_time_s': student_history['total_time'],
            'test2_mae_mV': student_ood['test2_mae_mV'],
            'test3_mae_mV': student_ood['test3_mae_mV'],
            'ood_avg_mV': student_ood['ood_avg_mV'],
            'alpha': args.alpha,
            'physics_params': student.get_physics_params(),
            'hybrid_params': student.get_hybrid_params(),
        },
    }

    results_file = output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Print summary table
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'Model':<20} {'Val MAE':>10} {'Test2':>10} {'Test3':>10} {'OOD Avg':>10}")
    print("-"*70)
    print(f"{'Teacher':<20} {results['teacher']['val_mae_mV']:>10.2f} {results['teacher']['test2_mae_mV']:>10.2f} {results['teacher']['test3_mae_mV']:>10.2f} {results['teacher']['ood_avg_mV']:>10.2f}")
    print(f"{'Student (a='+str(args.alpha)+')':<20} {results['student']['val_mae_mV']:>10.2f} {results['student']['test2_mae_mV']:>10.2f} {results['student']['test3_mae_mV']:>10.2f} {results['student']['ood_avg_mV']:>10.2f}")
    print("="*70)


def run_quick_test(args, device, output_dir):
    """Quick test: 5 epochs to verify everything works."""
    print("\n=== QUICK TEST MODE (5 epochs) ===\n")
    args.epochs = 5

    print("[1/4] Loading data...")
    train_loader, val_loader, stats = load_test4_training(
        data_dir=args.data_dir, device=device, batch_size=args.batch_size,
        verbose=True, seed=args.seed
    )

    print("[2/4] Training teacher (5 epochs)...")
    teacher = HybridPhysicsMLP(device=device)
    teacher, _ = train_teacher(
        model=teacher, train_loader=train_loader, val_loader=val_loader,
        stats=stats, epochs=5, lr=args.lr, patience=30,
        save_dir=output_dir, verbose=True
    )

    print("[3/4] Evaluating OOD...")
    evaluate_ood(teacher, device=device, verbose=True, data_dir=args.data_dir)

    print("[4/4] Testing student distillation (5 epochs)...")
    student = PhysicsHybrid12Param().to(device)
    train_student_distillation(
        student=student, teacher=teacher, train_loader=train_loader,
        val_loader=val_loader, alpha=0.1, epochs=5, lr=0.001, patience=40,
        verbose=True
    )

    print("\n=== QUICK TEST PASSED ===")


def run_teacher_only(args, device, output_dir):
    """Train only teacher model."""
    print("\n=== TEACHER-ONLY MODE ===\n")

    train_loader, val_loader, stats = load_test4_training(
        data_dir=args.data_dir, device=device, batch_size=args.batch_size,
        verbose=True, seed=args.seed
    )

    teacher = HybridPhysicsMLP(device=device)
    teacher, history = train_teacher(
        model=teacher, train_loader=train_loader, val_loader=val_loader,
        stats=stats, epochs=args.epochs, lr=args.lr, patience=30,
        save_dir=output_dir, verbose=True
    )

    teacher_ood = evaluate_ood(teacher, device=device, verbose=True, data_dir=args.data_dir)
    print(f"\nTeacher OOD Average: {teacher_ood['ood_avg_mV']:.2f} mV")


def run_inverse(args, device):
    """Run inverse solver: find max safe pressure or predict voltage."""
    from inverse import load_model, predict_voltage, PressureOptimizer

    checkpoint = args.checkpoint or str(Path(__file__).parent.parent.parent / 'results' / 'best_12param.pt')
    if not Path(checkpoint).exists():
        print(f"Error: Checkpoint not found: {checkpoint}", file=sys.stderr)
        sys.exit(1)

    model = load_model(checkpoint, device=device)
    optimizer = PressureOptimizer(model, device=device)

    if args.voltage is not None and args.current is not None and args.temperature is not None:
        # Find P_max
        result = optimizer.find_P_max(
            V_target=args.voltage, I=args.current, T=args.temperature,
            safety_margin_mV=args.safety_margin,
        )
        if args.json:
            print(json.dumps({
                'mode': 'inverse', 'P_max_bar': round(result.P_max, 2),
                'P_safe_bar': round(result.P_safe, 2),
                'V_achieved_V': round(result.V_achieved, 4),
                'converged': result.converged, 'iterations': result.iterations,
                'latency_ms': round(result.latency_ms, 2), 'method': result.method,
            }, indent=2))
        else:
            print(f"Maximum safe pressure: {result.P_safe:.1f} bar")
            print(f"  Theoretical max:     {result.P_max:.1f} bar")
            print(f"  Voltage at P_max:    {result.V_achieved:.4f} V")
            print(f"  Converged:           {result.converged}")
            print(f"  Latency:             {result.latency_ms:.1f} ms")

    elif args.current is not None and args.temperature is not None and args.pressure is not None:
        # Predict voltage
        V = predict_voltage(model, args.current, args.temperature, args.pressure, device=device)
        if args.json:
            print(json.dumps({
                'mode': 'predict', 'voltage_V': round(V, 4),
                'current_A': args.current, 'temperature_C': args.temperature,
                'pressure_bar': args.pressure,
            }, indent=2))
        else:
            print(f"Predicted voltage: {V:.4f} V")
            headroom = (1.85 - V) * 1000
            print(f"  Headroom to 1.85V:   {headroom:.0f} mV" if headroom > 0
                  else f"  OVER 1.85V by:       {abs(headroom):.0f} mV")
    else:
        print("Error: inverse mode requires --voltage --current --temperature (for P_max)")
        print("   or: --current --temperature --pressure (for voltage prediction)")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="PEM Electrolyzer PINN - MLOPS Demonstrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Quick test:     python main.py --mode quick-test
  Full training:  python main.py --mode full --epochs 100 --device cuda
  Teacher only:   python main.py --mode teacher-only --seed 42
  Ablation study: python main.py --mode ablation --device cuda
  Inverse solver: python main.py --mode inverse --voltage 1.85 --current 10 --temperature 75
        """
    )
    parser.add_argument("--mode", choices=["full", "quick-test", "teacher-only", "ablation", "inverse"],
                        default="full", help="Execution mode (default: full)")
    parser.add_argument("--data-dir", default="dataset/",
                        help="Path to dataset directory (default: dataset/)")
    parser.add_argument("--output-dir", default="results/",
                        help="Path to output directory (default: results/)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Distillation alpha: weight on labels vs teacher (default: 0.1)")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size (default: 4096)")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (default: 0.01)")
    parser.add_argument("--device", default="auto", help="Device: cuda/cpu/auto (default: auto)")

    # Inverse solver arguments
    parser.add_argument("--voltage", type=float, help="Target voltage [V] (for inverse mode)")
    parser.add_argument("--current", type=float, help="Current [A] (for inverse mode)")
    parser.add_argument("--temperature", type=float, help="Temperature [C] (for inverse mode)")
    parser.add_argument("--pressure", type=float, help="Pressure [bar] (for inverse predict)")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint path (for inverse mode)")
    parser.add_argument("--safety-margin", type=float, default=40.0, help="Safety margin [mV] (default: 40)")
    parser.add_argument("--json", action="store_true", help="Output as JSON (inverse mode)")

    args = parser.parse_args()

    # Setup device
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup seed
    set_seed(args.seed)

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("PEM Electrolyzer PINN - MLOPS Demonstrator")
    print("="*70)
    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Data dir: {args.data_dir}")
    print(f"Output dir: {args.output_dir}")

    # Dispatch
    if args.mode == "full":
        run_full(args, device, output_dir)
    elif args.mode == "quick-test":
        run_quick_test(args, device, output_dir)
    elif args.mode == "teacher-only":
        run_teacher_only(args, device, output_dir)
    elif args.mode == "ablation":
        from ablation import run_ablation
        run_ablation(args, device, output_dir)
    elif args.mode == "inverse":
        run_inverse(args, device)
        return

    print("\nDone!")


if __name__ == "__main__":
    main()

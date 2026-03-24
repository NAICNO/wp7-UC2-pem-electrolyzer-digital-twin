"""Plotting utilities for PEM electrolyzer PINN results."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # for non-interactive use


def plot_training_history(history, title='Training History', save_path=None):
    """Plot training and validation loss curves.

    Args:
        history: dict with keys 'train_loss' and 'val_loss' (lists of floats)
        title: plot title
        save_path: if provided, save figure to this path
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_voltage_prediction(actual, predicted, temperatures=None,
                           model_name='Model', save_path=None):
    """Plot actual vs predicted voltage with optional color coding.

    Args:
        actual: array of actual voltages
        predicted: array of predicted voltages
        temperatures: optional array for color coding
        model_name: name for title
        save_path: if provided, save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Time series comparison
    ax = axes[0]
    ax.plot(actual, 'b-', label='Actual', alpha=0.7, linewidth=1)
    ax.plot(predicted, 'r-', label='Predicted', alpha=0.7, linewidth=1)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Voltage (V)')
    ax.set_title(f'{model_name}: Voltage Prediction')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Scatter plot
    ax = axes[1]
    if temperatures is not None:
        sc = ax.scatter(actual, predicted, c=temperatures, cmap='coolwarm',
                       alpha=0.5, s=10)
        plt.colorbar(sc, ax=ax, label='Temperature (°C)')
    else:
        ax.scatter(actual, predicted, alpha=0.3, s=10, color='steelblue')

    # Perfect prediction line
    vmin = min(np.min(actual), np.min(predicted))
    vmax = max(np.max(actual), np.max(predicted))
    ax.plot([vmin, vmax], [vmin, vmax], 'k--', alpha=0.5, label='Perfect')
    ax.set_xlabel('Actual Voltage (V)')
    ax.set_ylabel('Predicted Voltage (V)')
    ax.set_title(f'{model_name}: Actual vs Predicted')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_ood_comparison(results_dict, save_path=None):
    """Bar chart comparing model OOD performance.

    Args:
        results_dict: dict of {model_name: {'test2_mae': float, 'test3_mae': float, 'val_mae': float}}
        save_path: if provided, save figure
    """
    models = list(results_dict.keys())
    test2 = [results_dict[m].get('test2_mae', 0) for m in models]
    test3 = [results_dict[m].get('test3_mae', 0) for m in models]
    val = [results_dict[m].get('val_mae', 0) for m in models]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, val, width, label='Validation MAE', color='steelblue')
    ax.bar(x, test2, width, label='Test2 MAE (current sweep)', color='coral')
    ax.bar(x + width, test3, width, label='Test3 MAE (pressure swap)', color='seagreen')

    ax.set_xlabel('Model')
    ax.set_ylabel('MAE (mV)')
    ax.set_title('Out-of-Distribution Generalization Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_physics_parameters(model, param_names=None, save_path=None):
    """Visualize learned physics parameters from a PINN model.

    Args:
        model: PyTorch model with physics parameters
        param_names: optional list of parameter names
        save_path: if provided, save figure
    """
    import torch

    # Extract physics parameters
    params = {}
    for name, param in model.named_parameters():
        if param.numel() == 1:  # scalar parameters (physics params)
            params[name] = param.item()

    if not params:
        print("No scalar physics parameters found.")
        return None

    names = list(params.keys())
    values = list(params.values())

    if param_names and len(param_names) == len(names):
        names = param_names

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.5)))
    y_pos = np.arange(len(names))
    colors = ['steelblue' if v >= 0 else 'coral' for v in values]
    ax.barh(y_pos, values, color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel('Parameter Value')
    ax.set_title('Learned Physics Parameters')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_residual_analysis(actual, predicted, model_name='Model', save_path=None):
    """Analyze prediction residuals.

    Args:
        actual: array of actual values
        predicted: array of predicted values
        model_name: name for title
        save_path: if provided, save figure
    """
    residuals = np.array(actual) - np.array(predicted)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Residuals over time
    axes[0].plot(residuals, alpha=0.5, linewidth=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Sample Index')
    axes[0].set_ylabel('Residual (V)')
    axes[0].set_title('Residuals Over Time')
    axes[0].grid(True, alpha=0.3)

    # Histogram
    axes[1].hist(residuals * 1000, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    axes[1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Residual (mV)')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Residual Distribution (MAE={np.mean(np.abs(residuals))*1000:.1f} mV)')
    axes[1].grid(True, alpha=0.3)

    # QQ-like plot (residual vs predicted)
    axes[2].scatter(predicted, residuals * 1000, alpha=0.3, s=5, color='steelblue')
    axes[2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Predicted Voltage (V)')
    axes[2].set_ylabel('Residual (mV)')
    axes[2].set_title('Residuals vs Predicted')
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f'{model_name}: Residual Analysis', fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def combined_analysis_plot(actual, predicted, temperatures, model_name='Model', save_path=None):
    """Combined 2x2 analysis plot (teleconnections-style).

    Args:
        actual: array of actual voltages
        predicted: array of predicted voltages
        temperatures: array of temperatures
        model_name: name for title
        save_path: if provided, save figure
    """
    residuals = np.array(actual) - np.array(predicted)
    mae = np.mean(np.abs(residuals)) * 1000

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Time series
    ax = axes[0, 0]
    ax.plot(actual, 'b-', label='Actual', alpha=0.7, linewidth=1)
    ax.plot(predicted, 'r-', label='Predicted', alpha=0.7, linewidth=1)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Voltage (V)')
    ax.set_title('Voltage Prediction')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top-right: Scatter
    ax = axes[0, 1]
    sc = ax.scatter(actual, predicted, c=temperatures, cmap='coolwarm', alpha=0.5, s=10)
    vmin, vmax = min(np.min(actual), np.min(predicted)), max(np.max(actual), np.max(predicted))
    ax.plot([vmin, vmax], [vmin, vmax], 'k--', alpha=0.5)
    ax.set_xlabel('Actual (V)')
    ax.set_ylabel('Predicted (V)')
    ax.set_title(f'Actual vs Predicted (MAE={mae:.1f} mV)')
    plt.colorbar(sc, ax=ax, label='Temp (°C)')
    ax.grid(True, alpha=0.3)

    # Bottom-left: Residual histogram
    ax = axes[1, 0]
    ax.hist(residuals * 1000, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(x=0, color='r', linestyle='--')
    ax.set_xlabel('Residual (mV)')
    ax.set_ylabel('Count')
    ax.set_title('Residual Distribution')
    ax.grid(True, alpha=0.3)

    # Bottom-right: Residual vs temperature
    ax = axes[1, 1]
    ax.scatter(temperatures, residuals * 1000, alpha=0.3, s=10, color='coral')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Residual (mV)')
    ax.set_title('Residuals vs Temperature')
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'{model_name} — Combined Analysis', fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig

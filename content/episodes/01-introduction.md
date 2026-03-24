# Introduction to PEM Electrolyzer PINNs

```{objectives}
- Understand what Physics-Informed Neural Networks (PINNs) are
- Learn about PEM electrolyzers and why voltage prediction matters
- Know the project objectives and repository structure
- Understand the 6-model comparison framework and the 12-number equation takeaway
```

```{admonition} Why This Matters
:class: tip

**The Scenario:** An operator at a green hydrogen plant notices the cell voltage climbing during a high-demand shift. She needs to know: *is it safe to increase pressure to boost hydrogen output, or will the membrane degrade?* A fast, accurate voltage model — one that works even at conditions the plant has never operated at — could answer that question in milliseconds.

**The Research Question:** Can a 12-parameter physics model, trained through knowledge distillation from a larger teacher network, predict cell voltage accurately enough to replace expensive CFD simulations — and generalize reliably to operating conditions never seen during training?

**What This Episode Gives You:** The big picture — how 6 models are compared, why physics-informed approaches win, and why the final model fits in a single equation with 12 numbers.
```

## Overview

This repository provides a complete framework for training physics-informed neural networks (PINNs) to predict PEM electrolyzer cell voltage from operating conditions (current, pressure, temperature).

PEM (Proton Exchange Membrane) electrolyzers split water into hydrogen and oxygen using electricity. Accurate voltage prediction is critical for optimizing efficiency, monitoring degradation, and designing control systems.

### Why Physics-Informed Neural Networks?

Traditional approaches have limitations:
- **First-principles models** require detailed knowledge of all physical processes, which is often incomplete
- **Pure ML models** fit training data well but fail on out-of-distribution (OOD) conditions

PINNs combine both: they encode electrochemical physics (Nernst equation, Butler-Volmer kinetics, Ohmic losses) while learning residual corrections from data.

## 6-Model Comparison Framework

The notebook trains **6 models** from scratch on the same data and compares their OOD generalization:

| Model | Type | Parameters | Physics? |
|-------|------|-----------|----------|
| PureMLP | Standard neural network | ~2,000 | No |
| BigMLP | Large neural network | ~50,000 | No |
| Transformer | Self-attention network | ~50,000 | No |
| Teacher (HybridPhysicsMLP) | Physics + MLP hybrid | ~9,000 | Partial |
| Pure Physics | 12-param electrochemistry (alpha=1.0) | 12 | Full |
| **Distilled Student** | 12-param electrochemistry (alpha=0.1) | **12** | **Full** |

The key finding: the 12-parameter distilled student achieves the best OOD generalization, beating neural networks with 50,000+ parameters.

```{figure} ../images/model_comparison.png
:alt: OOD performance bar chart and parameters vs OOD scatter
:width: 100%

Left: Out-of-distribution performance (lower is better). The 12-parameter student (17 mV) beats all neural networks. Right: Parameters vs OOD accuracy — the student sits in the ideal lower-left corner.
```

## Teacher-Student Architecture

This project uses a two-stage approach:

1. **Teacher Model** (HybridPhysicsMLP): 8 physics parameters + MLP residual (~9,354 parameters)
2. **Student Model** (PhysicsHybrid12Param): 12-parameter pure physics model, trained via knowledge distillation

The student learns from both real data and teacher predictions, achieving better OOD generalization with only 12 parameters.

## The Biggest Takeaway: A 12-Number Equation

After training, the distilled student is **not a neural network**. It is a deterministic algebraic equation defined by exactly 12 scalar constants:
- 6 physics parameters (exchange currents, resistance, transfer coefficients, limiting current)
- 6 hybrid correction parameters (logistic curve + pressure/temperature modulation)

Anyone with these 12 numbers can reproduce the model's predictions in any language (Python, MATLAB, Excel) without PyTorch or any ML framework. The notebook verifies this by implementing the full equation in pure NumPy and confirming it matches the PyTorch model to micro-volt precision.

## Self-Contained Repository

| Component | Location |
|-----------|----------|
| NORCE experimental data | `dataset/` (3 test datasets) |
| ML training scripts | `scripts/pem_electrolyzer/` |
| Interactive notebook | `demonstrator-v1.orchestrator.ipynb` |
| Sample results | `results/` |
| Dependencies | `requirements.txt` |

## Using AI Coding Assistants

If you're using an AI coding assistant, the repository includes an `AGENT.md` file with setup instructions. Tell your assistant:

> "Read AGENT.md and help me run the PEM electrolyzer PINN demonstrator on my NAIC VM."

## What You Will Learn

| Episode | Topic |
|---------|-------|
| 02 | Provisioning a NAIC VM |
| 03 | Setting up the environment |
| 04 | Understanding PEM electrolyzer data |
| 05 | PINN methodology and electrochemistry |
| 06 | Running experiments |
| 07 | Analyzing results |
| 08 | FAQ |

## Notebook Structure (11 Parts)

The interactive notebook covers the full pipeline:

| Part | Topic |
|------|-------|
| 1 | Environment Setup |
| 2 | Data Exploration |
| 3 | PINN Methodology |
| 4 | Training the Teacher |
| 5 | Baseline Models & Comparison (PureMLP, BigMLP, Transformer, Pure Physics) |
| 6 | Knowledge Distillation |
| 7 | Full 6-Model Comparison |
| 8 | Interactive Exploration (widgets) |
| 9 | Batch Ablation Study |
| 10 | Summary & Next Steps |
| 11 | **The Biggest Takeaway: 12-Number Equation** |

## Resources

- NAIC Portal: https://orchestrator.naic.no
- Repository: https://github.com/NAICNO/wp7-UC2-pem-electrolyzer-digital-twin

```{keypoints}
- PINNs combine electrochemical physics with ML for better generalization
- 6 models are compared: 3 pure-ML baselines + teacher + pure physics + distilled student
- The distilled student (12 params) beats 50,000-parameter neural networks on OOD
- After training, the student is a deterministic equation — reproducible from 12 numbers
- Knowledge distillation with alpha=0.1 transfers physics understanding from teacher to student
- All code and data are included in this repository
```

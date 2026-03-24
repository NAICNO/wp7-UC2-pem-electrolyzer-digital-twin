# Frequently Asked Questions

```{objectives}
- Find answers to common questions about the PEM PINN demonstrator
```

## Setup

**Q: What Python version is required?**
A: Python 3.8 or higher. The setup script will check this automatically.

**Q: Do I need a GPU?**
A: No, but training is much faster with a GPU. Quick test runs in under a minute on CPU. Full training (all 6 models) runs in a few minutes on GPU vs longer on CPU.

**Q: How do I check if my GPU is working?**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Q: I get an `AttributeError: total_mem` error. What do I do?**
A: This happens with PyTorch 2.10+ which renamed `total_mem` to `total_memory`. The setup script and notebook handle this automatically with a `getattr` fallback. Make sure you have the latest version (`git pull`).

## Training Details

**Q: Why different optimizers for teacher (SGD) vs student (Adam)?**
A: The teacher has ~9,000 parameters including an MLP — SGD with momentum prevents the MLP from overfitting to training-specific patterns, which is critical for OOD generalization. The student has only 12 constrained physics parameters — Adam's per-parameter adaptive learning rates help these sparse, heterogeneous parameters converge efficiently. Since the student has no MLP, the overfitting risk that SGD addresses does not apply.

**Q: What is ReduceLROnPlateau?**
A: A learning rate scheduler used for student training. It monitors validation MAE and halves the learning rate whenever the metric does not improve for 10 consecutive epochs (factor=0.5, patience=10). This is different from the teacher's CosineAnnealingLR, which follows a fixed cosine decay. ReduceLROnPlateau is better for the student because its 12-parameter landscape has unpredictable convergence speed.

**Q: Why is gradient clipping used?**
A: Both teacher and student use `clip_grad_norm_(max_norm=1.0)` to prevent exploding gradients. This is especially important for physics-constrained parameters that involve arcsinh and log transforms — these functions can produce very large gradients when the input is near boundary values (e.g., current near the limiting current).

**Q: Why does the student train for 200 epochs vs 100 for the teacher?**
A: The student uses ReduceLROnPlateau, which needs more epochs to explore the loss landscape — the learning rate only decreases when validation stalls, rather than on a fixed schedule. The student also has a longer early stopping patience (40 vs 30 epochs) because learning rate reductions can trigger delayed improvement.

**Q: How do I interpret the student's dual loss (L and D)?**
A: During student training, two losses are tracked: L (label loss) = MSE between student predictions and true voltage labels, and D (distillation loss) = MSE between student predictions and teacher predictions. With alpha=0.1, the total loss is 0.1 x L + 0.9 x D. Both should decrease during training. If D stays high while L decreases, the student is learning from data but not from the teacher — check that the teacher model loaded correctly.

**Q: What loss function is used?**
A: MSE (mean squared error) in Volts for training. MAE (mean absolute error) in millivolts for evaluation and comparison. The x1000 conversion from Volts to millivolts happens during validation and OOD evaluation, not during training. MAE is more interpretable for comparing models ("15 mV error" is clearer than "0.000225 MSE").

**Q: What are the two ablation studies?**
A: The CLI ablation (`--mode ablation`) compares **architectures** — it trains all 6 model types across 3 seeds to prove physics-informed models generalize better. The notebook ablation (Part 9) compares **hyperparameters** — it varies SGD vs Adam, keep-out vs random, learning rates, and weight decay across 3 seeds to show which training choices matter. They answer different questions and should both be reviewed. See Episode 06 for full details.

## Training

**Q: Why SGD instead of Adam?**
A: Adam adapts learning rates per-parameter, which tends to overfit to IID validation data. SGD with momentum provides smoother optimization and significantly better out-of-distribution (OOD) generalization. This is demonstrated in the ablation study (notebook Part 9).

**Q: Why keep-out validation instead of random split?**
A: Random splitting creates IID validation data, which doesn't test generalization. The temperature-based keep-out (76-80C) forces the model to interpolate to unseen conditions, which is critical for OOD performance.

**Q: What does alpha=0.1 mean in distillation?**
A: The distillation loss is: alpha x MSE(student, labels) + (1-alpha) x MSE(student, teacher). With alpha=0.1, the student learns 90% from the teacher and 10% from labels. This transfers the teacher's implicit physics understanding.

**Q: Why does the student outperform the teacher on OOD?**
A: The student has only 12 parameters, all grounded in physics equations. It cannot memorize training-specific patterns and must rely on physical relationships that generalize across operating conditions.

**Q: What is the difference between Pure Physics and Distilled Student?**
A: Both use the same 12-parameter architecture (PhysicsHybrid12Param). The difference is training:
- **Pure Physics** (alpha=1.0): trained on labels only, no teacher signal
- **Distilled Student** (alpha=0.1): trained 90% on teacher predictions, 10% on labels

The teacher's predictions act as a regularizer, leading to 5-10 mV better OOD performance for the distilled student.

**Q: Why are baselines (PureMLP, BigMLP, Transformer) included?**
A: To rigorously demonstrate that physics constraints help generalization. Without baselines, you cannot prove that the physics-informed approach is better than simply using a larger neural network. The 6-model comparison (notebook Part 7) shows that 50,000-parameter networks fail on OOD while the 12-parameter student succeeds.

**Q: Does the student use any neural network at inference?**
A: No. After training, the student is a deterministic algebraic equation with 12 fixed constants. There are no hidden layers, no weight matrices, no activation functions. It can be implemented in a spreadsheet. The notebook (Part 11) proves this by implementing the exact equation in pure NumPy and showing it matches the PyTorch model to micro-volt precision.

**Q: Can I reproduce the model without PyTorch?**
A: Yes. The notebook (Part 11) provides a standalone `predict_voltage_numpy()` function that uses only NumPy. The "Reproducibility Card" at the end lists all 12 constants. Copy the function and constants into any language (MATLAB, Excel, C++, Julia) to reproduce predictions exactly.

## Practical Deployment

**Q: How can the trained model be used in practice?**
A: The 12-parameter equation enables several applications:
- **Real-time monitoring**: Predict expected voltage from operating conditions and flag deviations as potential degradation
- **Anomaly detection**: Monitor the 12 learned parameters over time — drift in R_ohm or i0 values indicates membrane degradation or catalyst deactivation
- **Operating condition optimization**: Use gradient-based optimization on the closed-form equation to find current/pressure/temperature setpoints that minimize energy consumption
- **Transfer learning**: Retrain on data from a different electrolyzer to adapt the model, starting from the learned parameters as initialization

**Q: Why is R_ohm so high (~1.0 Ohm*cm2) compared to literature (0.05-0.20)?**
A: This electrolyzer uses **SWVF (Static Water Vapour Feed) technology**, where water vapor diffuses through a barrier membrane. This barrier adds ionic resistance not present in conventional liquid-fed PEM systems. The model correctly captures this technology-specific physics. See Episode 04 for details.

**Q: What are the limitations of this approach?**
A: The paper identifies these limitations:
- **Single system**: Validated on one NORCE electrolyzer — generalization to other systems needs verification
- **Steady-state only**: No dynamic or transient modeling (ramp-up, load changes)
- **No degradation**: Parameters are fixed after training — no time-dependent evolution
- **Single technology**: PEM only — not validated on alkaline or SOEC electrolyzers
- **Lumped parameters**: Cell-level effective values, not intrinsic catalyst properties

**Q: Can I use this model for a different electrolyzer?**
A: You would need to retrain on data from your electrolyzer. The architecture (PhysicsHybrid12Param) is general — the electrochemistry equations apply to any PEM system. The 12 parameter values are specific to this NORCE stack. Retraining on your data will produce new parameter values appropriate for your system.

## Data

**Q: What is the 50 cm2 active area?**
A: The electrolyzer has 4 cells x 12.5 cm2 each = 50 cm2 total. Current density (A/cm2) is calculated by dividing the measured current (A) by this area.

**Q: Why are H2 and O2 pressures separate?**
A: The Nernst equation requires separate partial pressures: V_nernst depends on ln(P_H2 x sqrt(P_O2)). Averaging them would lose important information about asymmetric pressure effects.

**Q: Is there a data leak between training and OOD evaluation?**
A: No. Training uses Test4 data only (loaded from `test4_subset.csv`). OOD evaluation uses Test2 and Test3 (separate CSV files from different NORCE test campaigns). Normalization statistics are computed from Test4 only. OOD results are used for reporting, not model selection.

**Q: Can I add my own data?**
A: Yes. Place CSV files in the `dataset/` directory with the same 5 columns (PS-I-MON, H-P1, O-P1, T-ELY-CH1, CV-mean). Use `--data-dir` to point to your data.

## Troubleshooting

**Q: CUDA out of memory error**
A: Reduce batch size: `--batch-size 1024` or `--batch-size 512`

**Q: ModuleNotFoundError**
A: Make sure you're in the virtual environment: `source venv/bin/activate`

**Q: Permission denied on setup.sh**
A: Run `chmod +x setup.sh vm-init.sh`

**Q: Widget ImportError in the notebook**
A: Make sure you have the latest code (`git pull`). The notebook imports `build_widgets` and `create_execution_mode_dropdown` from `widgets.py`. If you see errors about `create_model_widgets`, you have an old version.

**Q: Git pull fails because notebook has local changes**
A: Running the notebook adds cell outputs, which Git tracks. Reset and pull:
```bash
git checkout -- demonstrator-v1.orchestrator.ipynb
git pull
```

```{keypoints}
- GPU recommended but not required
- SGD for teacher (prevents MLP overfitting), Adam for student (helps sparse physics params)
- CosineAnnealingLR for teacher, ReduceLROnPlateau for student
- Gradient clipping (max_norm=1.0) stabilizes physics-constrained training
- MSE for training, MAE in mV for evaluation
- Keep-out validation is the most impactful training choice for OOD
- Two ablation studies: CLI compares architectures, notebook compares hyperparameters
- 6 models are compared to validate the physics approach
- The distilled student (12 params) beats 50K-param baselines on OOD
- After training, the student is a deterministic equation (no neural network)
- The model is reproducible from 12 numbers in any language
- No data leak: Test4 for training, Test2/Test3 for evaluation (separate files)
- 50 cm2 active area (4 cells x 12.5 cm2)
```

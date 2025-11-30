# Hyperparameter Search Guide for Vision Transformer

## Three Approaches Comparison

### 1. **Optuna + MLflow** (RECOMMENDED) ⭐

**File:** `hyperparameter_search.py`

**Pros:**

- ✅ **Smart search**: Learns from previous trials (Bayesian optimization)
- ✅ **Efficient**: Prunes bad trials early to save time
- ✅ **Fewer trials needed**: 20-30 trials often enough
- ✅ **Industry standard**: Used by top ML teams
- ✅ **Built-in visualization**: Optuna provides optimization plots

**Cons:**

- ❌ Requires `optuna` package: `pip install optuna`
- ❌ Slightly more complex setup

**When to use:**

- You have limited compute budget
- Want best results with fewer trials
- Need to explore large hyperparameter spaces

**Estimated time:** 15-20 hours for 20 trials × 30 epochs

**How to run:**

```bash
pip install optuna
python hyperparameter_search.py
```

---

### 2. **Grid Search + MLflow**

**File:** `grid_search.py`

**Pros:**

- ✅ **Simple**: Easy to understand and implement
- ✅ **Exhaustive**: Tests all combinations
- ✅ **Reproducible**: Same configs every time
- ✅ **No extra dependencies**: Just MLflow

**Cons:**

- ❌ **Slow**: Must test all combinations
- ❌ **Exponential growth**: 5 params × 3 values each = 243 trials!
- ❌ **Wastes compute**: Tests bad combinations

**When to use:**

- Small hyperparameter space (< 20 combinations)
- Need exhaustive search for paper/research
- Have unlimited compute budget

**Estimated time:** 10-30 hours depending on grid size

**How to run:**

```bash
python grid_search.py
```

---

### 3. **Manual Priority Configs** (Your current approach)

**File:** `vision_transformer_training.py` (priority_configs)

**Pros:**

- ✅ **Quick start**: Based on expert knowledge
- ✅ **Targeted**: Only test promising configs
- ✅ **No dependencies**: Works immediately

**Cons:**

- ❌ **Manual selection**: Requires domain expertise
- ❌ **Limited exploration**: Might miss optimal configs
- ❌ **No learning**: Each trial independent

**When to use:**

- Quick validation of known-good configs
- Limited time/budget
- Have strong intuition about good hyperparameters

**Estimated time:** 3-4 hours for 5 configs × 30 epochs

---

## Recommended Workflow

### Stage 1: Quick Validation (1-2 hours)

```python
# Test priority configs first
priority_configs = [
    {"learning_rate": 1e-3, "weight_decay": 0.1, "warmup_epochs": 10, "dropout_rate": 0.15},
    {"learning_rate": 2e-3, "weight_decay": 0.15, "warmup_epochs": 5, "dropout_rate": 0.2},
]
# Run these manually to verify setup works
```

### Stage 2: Optuna Search (15-20 hours)

```bash
# Run smart search to find best hyperparameters
python hyperparameter_search.py
```

### Stage 3: Final Training (2-3 hours)

```python
# Take best config from Optuna and train for 100 epochs
# This is your final model for evaluation
```

---

## Installation

```bash
# Required
pip install mlflow torch torchvision torchmetrics

# For Optuna (recommended)
pip install optuna

# Optional: For advanced visualization
pip install optuna-dashboard
optuna-dashboard optuna_study.pkl  # View study in browser
```

---

## Viewing Results in MLflow

### In Databricks:

1. Go to **Experiments** in left sidebar
2. Find your experiment: `/Users/colizu2020@gmail.com/cifar-100-vit-optuna`
3. Click on parent run to see all trials
4. Use **Compare** to see parallel coordinates plot
5. Sort by `best_val_accuracy` to find top performers

### Locally:

```bash
mlflow ui
# Open browser to http://localhost:5000
```

---

## Interpreting Optuna Results

After running Optuna, you'll get:

### 1. Best Trial

```
Best trial: 12
Best validation accuracy: 0.6234
Best hyperparameters:
  learning_rate: 0.00147
  weight_decay: 0.0823
  warmup_epochs: 8
  dropout_rate: 0.167
  label_smoothing: 0.083
```

### 2. Optimization History Plot

Shows how validation accuracy improves over trials

### 3. Parameter Importance

Shows which hyperparameters matter most:

```
learning_rate: 45%
weight_decay: 23%
dropout_rate: 18%
warmup_epochs: 10%
label_smoothing: 4%
```

### 4. Parallel Coordinate Plot

Visualize relationship between all hyperparameters and accuracy

---

## Advanced: Ray Tune (For Distributed Search)

If you have access to multiple GPUs or a cluster:

```bash
pip install ray[tune]
```

**Benefits:**

- Run multiple trials in parallel
- Automatic checkpointing
- Population-based training
- ASHA scheduler for early stopping

**File:** `ray_tune_search.py` (create if needed)

---

## Hyperparameter Ranges (Based on ViT Research)

| Parameter       | Min  | Max  | Best Starting Point | Impact      |
| --------------- | ---- | ---- | ------------------- | ----------- |
| learning_rate   | 1e-4 | 5e-3 | 1e-3                | High ⭐⭐⭐ |
| weight_decay    | 0.01 | 0.2  | 0.1                 | High ⭐⭐⭐ |
| warmup_epochs   | 5    | 20   | 10                  | Medium ⭐⭐ |
| dropout_rate    | 0.05 | 0.3  | 0.15                | Medium ⭐⭐ |
| label_smoothing | 0.0  | 0.2  | 0.1                 | Low ⭐      |

Focus on `learning_rate` and `weight_decay` first - they have the biggest impact!

---

## Cost-Benefit Analysis

Assuming Google Colab T4 GPU:

| Approach           | Trials | Hours | Cost | Expected Accuracy |
| ------------------ | ------ | ----- | ---- | ----------------- |
| Priority Configs   | 5      | 3-4   | Free | 58-62%            |
| Optuna             | 20     | 15-20 | Free | 62-68%            |
| Grid Search        | 50     | 30-40 | ~$10 | 62-68%            |
| Optuna (extensive) | 50     | 40-50 | ~$15 | 65-70%            |

**Verdict:** Optuna with 20 trials gives best accuracy-per-hour ratio!

---

## Tips for Success

1. **Start Small**: Run 3-5 trials to verify setup works
2. **Monitor Early**: Check first trial completes successfully
3. **Use Pruning**: Optuna's MedianPruner stops bad trials at epoch 10
4. **Save Studies**: `joblib.dump(study)` to resume later
5. **Check GPU Usage**: `nvidia-smi` should show high utilization
6. **Track Costs**: Set MLflow tags with GPU type and cost estimates

---

## Troubleshooting

### "CUDA out of memory"

- Reduce batch size to 128 or 192
- Add `torch.cuda.empty_cache()` after each trial

### "MLflow experiment not found"

```python
mlflow.set_experiment(experiment_name)  # Creates if doesn't exist
```

### Optuna trials fail silently

```python
# Add error logging
try:
    study.optimize(objective, n_trials=20)
except Exception as e:
    print(f"Trial failed: {e}")
    import traceback
    traceback.print_exc()
```

### Grid search takes too long

- Reduce combinations: test 2 values per param instead of 3
- Reduce epochs: use 20 epochs instead of 30 for initial search

---

## Next Steps

1. ✅ Run `hyperparameter_search.py` with 20 trials
2. ✅ Identify top 3 configurations from MLflow UI
3. ✅ Train each for 100 epochs to get final performance
4. ✅ Pick best model and evaluate on test set
5. ✅ Document best config in `notes.md`

Good luck! 🚀

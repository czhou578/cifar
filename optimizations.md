Adam vs AdamW:

Cosine Annealing: CosineAnnealingWarmRestarts -> Includes periodic restarts; the learning rate resets after each cycle according to the cosine curve.

Learning Rate with weight decay: Just L2 regularization. Penalizes very heavy weights and tries to
prevent overfitting. This penalty discourages the model from relying too heavily on any single input feature, promoting smoother weight distributions and reducing the risk of fitting to noise in the training data

Momentum: smooths out gradient oscillations, helps escape local minima, accelerates convergence in consistent gradient directions

Gradient clipping: prevent exploding gradients, less sensitive to learning rate choice, more consistent gradient updates

Gradient accumulation: Larger batch size, improved convergence with memory efficiency.

Guidelines for Tuning:

1. learning rate
2. momentum and mini batch size, hidden units
3. layers, learning rate decay
4.

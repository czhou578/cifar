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

Dropout: It works by randomly setting a fraction of the input units to zero during training, which helps to prevent complex co-adaptations on the training data.

Label Smoothing: Addresses issue of overconfidence in neural networks. Can improve generalization, acts as regularization, and better calibration.

But... can potentially slow convergence, loss of interpretability, not always beneficial for datasets with clean and well-separated classes.

Consider quantization (e.g., INT8) for CPU-based inference to reduce size and execution time.

torch.inference_mode():

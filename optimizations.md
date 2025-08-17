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

torch.inference_mode()

**Model Fusing**: merges the mathematical operations of these layers into a single layer. Should be done after traininga and before eval.

**Pruning**:

prune.ln_structured: This function is the core of the pruning logic.

module: The layer to prune (e.g., a Conv2d layer).
name='weight': Specifies that we are pruning the weight tensor of the layer.
amount=0.4: This tells PyTorch to remove 40% of the structures.
n=2: Uses the L2-norm to measure the importance of each filter. Filters with a smaller L2-norm are considered less important and are removed first.
dim=0: For a Conv2d weight tensor with shape [out_channels, in_channels, kH, kW], dim=0 means we are pruning entire filters (i.e., along the out_channels dimension).

**Channel Last Memory Format**

It reorganizes the memory layout of tensors from (N, C, H, W) to (N, H, W, C), which allows PyTorch's underlying libraries (like oneDNN) to use much more efficient algorithms.

Pin memory has no effect on CPU

Gradient Accumulation for effective larger batch sizes: This simulates larger batch sizes without memory overhead, leading to better convergence and fewer optimizer steps:

optimizer.zero_grad(set_to_none=True): More memory efficient then default gradient zeroing
prefetch_factor = 4 -> prefetch more batches

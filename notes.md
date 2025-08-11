Adagrad:

Accumulates all past gradients forever
Learning rate monotonically decreases over time
Can become too slow or stop learning entirely
Good for sparse gradients/features

Adam:

Uses exponential moving averages of gradients
Maintains adaptive learning rates that don't vanish
Combines momentum + adaptive learning rates
Better convergence properties

CrossEntropyLoss is the standard choice for multi-class classification as it:

Handles probability distributions across classes
Works with integer class labels (0-99 for CIFAR-100)
Includes softmax activation internally

Pytorch Autograd Engine Architecture

1. Function (base class for all diff. operations), Node, Edge (connects nodes)
2. Topological sorting of the graph when execute is called
3. Backward pass: top sort, ready queue, executes backward functions in parallel, and accumulates gradients

Memory Optimizations:

Saved Tensor Hooks: Reduce memory usage during backward pass
Graph Pruning: Remove unnecessary nodes
In-place Operations: Modify tensors without creating new nodes

Building CNN backbone would improve NN:

- preserve spatial structure and learn local patterns (build up to complex features)
- less parameters and less prone to overfitting

**CNN**

- how many filters do we want? Filter is a small matrix that we define the dimensions (init with random nums)

- Out channels is a hyperparameter -> increase channels with depth. Deeper layers use more channels
- Padding in conv2d preserves spatial dimensions since CNN shrinks output size. Also improves feature detection at borders.

**2D Batch Norm**

- How it works: For each channel in the feature map, it calculates the mean and standard deviation across all the examples in the current mini-batch. It then normalizes the activations to have a mean of 0 and a standard deviation of 1. It also has learnable parameters to scale and shift the result. This helps to stabilize and accelerate the training process.

---

The nn.Module base class has a special **setattr** method. This method is called every time you do self.some_attribute = value. It checks what value is:

If value is an nn.Parameter: It directly adds this tensor to an internal dictionary of parameters (e.g., \_parameters). This is how a module knows this specific tensor needs to have its gradient calculated.
If value is another nn.Module (like a Conv2d or Linear layer): It adds the submodule to an internal dictionary of modules (e.g., \_modules).

---

The key insight: ReLU decides what constitutes a "feature detection" (positive values), then MaxPool finds the strongest detection among those actual detections.

Things to add later:

Fuse modalities: Create a custom nn.Module for multimodal fusion (e.g., attention-based concatenation).
Handle dynamic shapes: Ensure model handles variable text lengths with padding/masking.
Why not add multiple layers to the nn.Sequential?

num_workers = 4 for dataloaders

```python
python -m torch.utils.bottleneck model.py
```

**Parameter Registration**

- assigns parameters to an attribute of your module, Pytorch registers it automatically.

Benefits of Parameter Registration
Automatic Optimization: Optimizers can find all parameters via .parameters()
Device Movement: .to(device) moves all registered parameters
State Management: .state_dict() saves/loads all registered parameters
Gradient Tracking: All registered parameters participate in backpropagation

**Forward Hooks**

- Forward hooks are callback functions that execute during the forward pass, allowing you to inspect or modify intermediate computations.

**Pin Memory**

- Pin memory refers to allocating memory in a way that prevents the operating system from swapping it to disk (paging). Pinned memory is a limited system resource. Using too much can cause system instability.

**Persistent Workers**

- keep DataLoader worker processes alive between epochs instead of recreating them. This eliminates creation of new processes every cpoch and is memory efficient.

**Ablation Tests For Loss**

2.82 - Adam optim, lr 1e-3, 5 epochs, 2 CNN layers with Batchnorm
2.765 - SGD with momentum, lr = 0.01
2.80 - SGD with momentum, lr = 0.01 and weight decay of 5e-4.
2.8445 - AdamW with lr = 1e-3
3.1575 - AdamW with lr = 0.0001
2.06 - AdamW with 30 epochs

**Multimodal Contrastive Loss**

The core principle involves bringing similar instances (positive pairs) closer together in an embedding space while pushing dissimilar instances (negative pairs) farther apart. This is achieved by defining a loss function that penalizes the model when the distance between positive pairs exceeds a predefined margin threshold, thereby encouraging the model to capture semantic relationships between modalities.

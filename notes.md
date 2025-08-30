**Adagrad**:

Accumulates all past gradients forever
Learning rate monotonically decreases over time
Can become too slow or stop learning entirely
Good for sparse gradients/features

**AdamW**:

Uses exponential moving averages of gradients
Maintains adaptive learning rates that don't vanish
Combines momentum + adaptive learning rates
Better convergence properties

**Weight Decay**
https://towardsdatascience.com/weight-decay-and-its-peculiar-effects-66e0aee3e7b8/

## Why use for CIFAR 100?

- Instead of incorporating the penalty into the loss, AdamW applies weight decay directly to the parameters after the adaptive gradient update, ensuring that the regularization is not influenced by the adaptive learning rate dynamics.

CrossEntropyLoss is the standard choice for multi-class classification.

## Label Smoothing

**label smoothing**: Whenever a classification neural network suffers from overfitting and/or overconfidence, we can try label smoothing. It restrains the largest logit from becoming much bigger than the rest.

For example, [1, 0, 0] may become [0.9, 0.5, 0.5]

**When to use it**:
Reduces overfitting, enhances model calibration, boosts robustness

**When not to use it**:

If your image labels are highly accurate and noise-free, smoothing might unnecessarily introduce artificial uncertainty, potentially harming accuracy or slowing convergence without benefits.

In tasks where the model needs to abstain from predicting on uncertain inputs (e.g., rejecting low-confidence samples), smoothing can worsen this by making confident predictions less distinguishable from uncertain ones.

Handles probability distributions across classes
Works with integer class labels (0-99 for CIFAR-100)
Includes softmax activation internally

## Pytorch Autograd Engine Architecture

1. Function (base class for all diff. operations), Node, Edge (connects nodes)
2. Topological sorting of the graph when execute is called
3. Backward pass: top sort, ready queue, executes backward functions in parallel, and accumulates gradients

Memory Optimizations:

Saved Tensor Hooks: Reduce memory usage during backward pass
Graph Pruning: Remove unnecessary nodes
In-place Operations: Modify tensors without creating new nodes

**CNN**

- how many filters do we want? Filter is a small matrix that we define the dimensions (init with random nums)

- Out channels is a hyperparameter -> increase channels with depth. Deeper layers use more channels

- Padding in conv2d preserves spatial dimensions since CNN shrinks output size. Also improves feature detection at borders.

**2D Batch Norm**

- How it works: For each channel in the feature map, it calculates the mean and standard deviation across all the examples (channels, in this case, each of RGB) in the current mini-batch. It then normalizes the activations to have a mean of 0 and a standard deviation of 1. It also has learnable parameters to scale and shift the result. This helps to stabilize and accelerate the training process.

**Alternatives to 2D Batch Norm**:

GroupNorm: calculates the mean and variance and normalizes across a group (splits the C channels into G groups (you choose G, like 32).)

This makes GroupNorm "batch-independent"—it doesn't care how big your batch is, because stats are calculated per example and per group.

Benefits:

- good for distributed training because doesn't require syncing states

- handles small batches well, since computes states within channel groups per sample.

### How are parameters set in the MLP class as shown in the code?

The nn.Module base class has a special **setattr** method. This method is called every time you do self.some_attribute = value. It checks what value is:

If value is an nn.Parameter: It directly adds this tensor to an internal dictionary of parameters (e.g., \_parameters). This is how a module knows this specific tensor needs to have its gradient calculated.
If value is another nn.Module (like a Conv2d or Linear layer): It adds the submodule to an internal dictionary of modules (e.g., \_modules).

## nn.Conv2d

This is PyTorch's implementation of a convolutional layer. The size of your input data doesn't matter!

1. in_channels: how many features are we passing in. Our features are our colour bands, in greyscale, we have 1 feature, in colour, we have 3 channels.

2. out_channels: how many kernels do we want to use. Analogous to the number of hidden nodes in a hidden layer of a fully connected network.

3. kernel_size: the size of the kernel. Above we were using 3x3. Common sizes are 3x3, 5x5, 7x7.

4. stride: the “step-size” of the kernel.

5. padding: the number of pixels we should pad to the outside of the image so we can get edge pixels.

## nn.MaxPool2d

1. aggregate the data, usually using the maximum or average of a window of pixels.

```python
nn.MaxPool2d(2) # 2x2 kernel, moving 2 pixels at a time
```

## nn.Dropout

1. Generally, use a small dropout value of 20%-50% of neurons, with 20% providing a good starting point.

2. Use a large learning rate with decay and a large momentum. Increase your learning rate by a factor of 10 to 100 and use a high momentum value of 0.9 or 0.99.

3. Use a larger network

The key insight: ReLU decides what constitutes a "feature detection" (positive values), then MaxPool finds the strongest detection among those actual detections.

```python
python -m torch.utils.bottleneck model.py
```

# Journey of a Matrix Transformation Through the Layers

## First Conv layer

1. Start batch shape is (256, 3, 3, 32) # 256 images per batch
2. Conv1_1: 3 channels in so (256, 64, 3, 32)
3. x = bn_1(x) and relu -> doesn't change shape

4. conv1_2: (256, 64, 3, 2)
5. x = bn_1(x) and relu -> doesn't change shape

6. maxpool1 -> (256, 64, 16, 16)
7. dropout doesn't change shape

## Second Conv layer

8. conv2_1 -> (256, 128, 16, 16)
9. x = bn_1(x) and relu -> doesn't change shape
10. conv2_2 -> (256, 128, 16, 16)
11. x = bn_1(x) and relu -> doesn't change shape
12. maxpool2 -> (256, 128, 8, 8)

## Third Conv Layer

13. conv3_1 -> (256, 256, 16, 16)
14. x = bn_1(x) and relu -> doesn't change shape
15. conv2_2 -> (256, 256, 16, 16)
16. x = bn_1(x) and relu -> doesn't change shape
17. maxpool2 -> (256, 256, 4, 4)

## Fourth Conv Layer

18. conv2_1 -> (256, 512, 16, 16)
19. x = bn_1(x) and relu -> doesn't change shape
20. conv2_2 -> (256, 512, 16, 16)
21. x = bn_1(x) and relu -> doesn't change shape
22. maxpool2 -> (256, 512, 2, 2)

## Flatten

23. (256, 512 \* 4) -> (256, 2048)

## Classification Head

24. linear -> (256, 1024)
25. Relu and dropout changes nothing
26. linear -> (256, 512)
27. Relu and dropout changes nothing
28. linear -> (256, 100)

End result is for each image, its classification among the 100 classes in CIFAR

**Parameter Registration**

- assigns parameters to an attribute of your module, Pytorch registers it automatically.

```python
('fc1', nn.Linear(512 * 2 * 2, 1024)),

# What happens internally in Linear
class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Automatically registered as parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
```

Benefits of Parameter Registration
Automatic Optimization: Optimizers can find all parameters via .parameters()
Device Movement: .to(device) moves all registered parameters
State Management: .state_dict() saves/loads all registered parameters
Gradient Tracking: All registered parameters participate in backpropagation

**Forward Hooks**

- Forward hooks are callback functions that execute during the forward pass, allowing you to inspect or modify intermediate computations.

**Dataloader**

- Pin memory refers to allocating memory in a way that prevents the operating system from swapping it to disk (paging). Pinned memory is a limited system resource. Using too much can cause system instability.

- Prefetch Fector controls the number of batches loaded in advance by each worker process to improve data loading efficiency

- keep DataLoader worker processes alive between epochs instead of recreating them. This eliminates creation of new processes every epoch and is memory efficient.

**Ablation Tests For Loss**

2.82 - Adam optim, lr 1e-3, 5 epochs, 2 CNN layers with Batchnorm
2.765 - SGD with momentum, lr = 0.01
2.80 - SGD with momentum, lr = 0.01 and weight decay of 5e-4.
2.8445 - AdamW with lr = 1e-3
3.1575 - AdamW with lr = 0.0001
2.06 - AdamW with 30 epochs

**Multimodal Contrastive Loss**

The core principle involves bringing similar instances (positive pairs) closer together in an embedding space while pushing dissimilar instances (negative pairs) farther apart. This is achieved by defining a loss function that penalizes the model when the distance between positive pairs exceeds a predefined margin threshold, thereby encouraging the model to capture semantic relationships between modalities.

**Accumulation Steps**

- accumulate gradients from multiple batches and then update once.

Benefits of large batches:

- less noisy gradient estimates
- smoother loss curves, and generalizes better
- Batchnorm computes statistics on the smaller batches, but not effective batch.

## GPU Training Optimizations

**Fuse Model**

Combines multiple layer operations into a single kernel, instead of multiple intermediary memory allocations.

In my case, fusing the conv, batch, and relu operations into one computation unit, saving memory and time, using `torch.quantization.fuse_modules` in eval mode ONLY

Reduces the number of memory transfers between GPU and memory, and less overhead per operation. Quantized operations are much faster when fused.

1. The Effect of Fusing:

- Fusing bakes the BatchNorm parameters (the learned gamma and beta along with the final running_mean and running_var) directly into the weights and bias of the preceding Conv2d layer.

2. What if you fuse during training?

- You would destroy the dynamic, per-batch normalization. The BatchNorm layer's stabilizing effect would be gone, and the model would be trying to learn with a fixed, stale normalization that doesn't adapt to each new batch. This would severely destabilize training and prevent the model from converging.

- Fusion "folds" BN into Conv, fixing the stats and turning BN into a static affine transform. You lose the regularization, making the model more prone to overfitting or vanishing/exploding gradients

Layers have to be consecutive.

```python
   torch.quantization.fuse_modules
```

## Torch.backends library:

.cudnn.benchmark

- First few iterations, cuDNN measures performance for your specific setup. Caches the fastest algorithm found.

* Disable when input sizes are variable, or training runs are short.

.cuda.matmul.allow_tf32

- TF32 is different then FP32 since it is same range but reduced precision.

* Enable on newest GPU, training when speed > precision.

.cudnn.allow_tf32

- Allows cuDNN operations (convolutions, etc.) to use TF32 format on supported hardware. Controls convolutions, pooling, normalization.

Fastest: All optimizations enabled
↓ benchmark=True + allow_tf32=True
↓ benchmark=True + allow_tf32=False  
 ↓ benchmark=False + allow_tf32=True
Slowest: benchmark=False + allow_tf32=False

Find maximum batch size dynamically:

```python

def find_max_batch_size(model, device, input_shape=(3, 32, 32), max_attempts=10):
    model.eval()  # Inference mode for memory estimate
    batch_size = 128  # Start low
    for _ in range(max_attempts):
        try:
            with torch.no_grad(), autocast(device_type='cuda'):
                dummy_input = torch.randn(batch_size * 2, *input_shape).to(device)  # Test double to probe
                output = model(dummy_input)
            batch_size *= 2  # Double if successful
        except RuntimeError as e:  # Catch OOM
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                break
            else:
                raise e
    max_batch = batch_size // 2  # Last successful
    safe_batch = int(max_batch * 0.9)  # 10% headroom for gradients/training
    print(f"Max probed batch_size: {max_batch}, Using safe: {safe_batch}")
    return safe_batch

# Call: batch_size = find_max_batch_size(mlp, device)
# Then use in train_loader
```

**When should I use autocast and mixed precision?**

- Training Large Models (like your CNN with 4 conv blocks)

Your model has significant memory requirements
You want to fit larger batch sizes in GPU memory
Limited GPU Memory

You're hitting CUDA out-of-memory errors
Want to increase batch size from 256 → 512+ for better training

Modern GPUs (V100, A100, RTX 30/40 series) have Tensor Cores
Can get 1.5-2x training speedup with minimal accuracy loss
Stable Training Setup (which you have)

Using proper loss scaling (GradScaler)
Well-tested model architecture
Sufficient regularization (BatchNorm, Dropout)

**OneCycleLR**

- prevents early gradient explosions with large models
- Escapes local minima quickly, accelerates training
- Low LR at end.

Scale LR based on effective batch size.

max_lr: choice is critical - too high causes instability
pct_start: affects training dynamics

Aggressive schedules like OneCycleLR might overfit to the validation set if not paired with strong regularization (e.g., your Dropout or augmentations).

**Autograd loss backward prop**

Use CrossEntropyLoss / BCEWithLogitsLoss instead of manual softmax/sigmoid + log.
Avoid in-place ops on tensors needed for grad.
Zero grads before backward (optimizer.zero_grad(set_to_none=True)).
Use GradScaler + autocast for mixed precision.
Clip gradients if exploding.
Monitor gradient norms and parameters for NaN/Inf.
Use torch.autograd.set_detect_anomaly(True) for debugging.
If writing custom autograd, verify with gradcheck in double precision.

**Torch Metrics**

Accuracy: Example: If your CIFAR-100 model makes 1000 predictions and gets 850 correct:
Accuracy = 850/1000 = 0.85 or 85%

Precision: "When the model says it's a dog, how often is it actually a dog?"

Model predicts 100 images as "airplane"
80 are actually airplanes, 20 are misclassified (maybe cars or birds)

------> Precision = 80/100 = 0.80 or 80%

**Recall:** Of all the actual dogs in the dataset, how many did the model find?

There are 120 actual airplane images in the test set
Model correctly identifies 80 of them as airplanes
Model misses 40 airplanes (classifies them as something else)
-------> Recall = 80/120 = 0.67 or 67%

**F1:**

When you need to balance precision and recall
When classes are imbalanced
When both false positives and false negatives are important

High Precision, Low Recall: Conservative model - when it predicts a class, it's usually right, but it misses many instances
Low Precision, High Recall: Liberal model - catches most instances but makes many false predictions
Balanced F1: Good compromise between precision and recall
Accuracy vs F1:
Accuracy can be misleading with imbalanced data
F1 provides better insight into per-class performance

| Stage              | Recommended Practice                                                |
| ------------------ | ------------------------------------------------------------------- |
| Mode Setting       | `model.eval()`, and `model.train()` where appropriate               |
| Gradient Handling  | Wrap inference in `torch.inference_mode()` or `torch.no_grad()`     |
| Precision & Format | Use FP16 or quantization depending on deployment scenario           |
| Optimization       | Leverage TorchScript, ONNX, TensorRT, fast loaders, bucketing, etc. |
| Memory Management  | Clean variables and GPU cache post-inference                        |

Epoch 50 finished
Training - Loss: 1.4756, Accuracy: 0.5858
Validation - Loss: 2.0236, Accuracy: 0.4848
Validation - Precision: 0.4847, Recall: 0.4877, F1: 0.4813
Training has completed

1. Why It Breaks Training
   The core of the issue lies in how BatchNorm2d works.

During Training (model.train() mode): BatchNorm is a dynamic layer. It normalizes the output of the convolution layer using the mean and standard deviation of the current mini-batch. It also updates its internal running_mean and running_var statistics with a moving average. This per-batch normalization is crucial for stabilizing the learning process, reducing internal covariate shift, and allowing for higher learning rates.

The Opportunity for Fusion: Since a Conv2d layer is also a linear operation, and an eval() mode BatchNorm2d is another linear operation, their math can be merged. You can pre-calculate a new set of weights and a new bias for the Conv2d layer that produces the exact same output as the original Conv2d followed by the BatchNorm2d.

```python3
torch.quantization.fuse_modules
```

**Quantization for INT8**

It converts the model's weights and activations from 32-bit floating-point numbers (FP32) to 8-bit integers (INT8). Sacrifices accuracy for faster training.

**Pruning**

Pruning is an excellent technique to reduce model size and potentially speed up inference by removing less important weights. Structured pruning, which removes entire filters or channels, is particularly effective because it creates a smaller, dense model that doesn't require special hardware for a speedup.

Train a model to convergence (which you have already done).
Prune the trained model by removing a certain percentage of weights.
Fine-tune the pruned model for a few epochs to recover the accuracy lost during pruning.
Evaluate the final, smaller model.

how is new max learning rate calculated?

51% CPU result with 20 epochs and gradient accumulation

prefetch factor and persistent_workers only works when num workers > 0

An asynchronous context manager in Python is an object that allows for the allocation and release of resources within asynchronous code, ensuring reliable setup and teardown logic even if the asynchronous operations encounter errors or interruptions.

RandAugment: better and stronger transforms from torchvision.

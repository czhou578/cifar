Volume: 32 x 32 x 3 where the 3 is considered the "depth" of a volume.
Filter must have same depth as the volume

32 x 32 x 3 image with 5 x 5 filter will result in 28 x 28 activation map

Fully connected layer:

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

**Weight Initialization**

If weights too small, then gradient update will be too small. Opposite if weights are too big.

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

Examples in the batch are coupled mathematically, so its non deterministic.
The centering operation is differentiable

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

Do this to reduce number of parameters and reduce overfitting. Does this to every layer of the volume spatially, in depth slice.

1. aggregate the data, usually using the maximum or average of a window of pixels.

```python
nn.MaxPool2d(2) # 2x2 kernel, moving 2 pixels at a time
```

## nn.RELU

- Think of ReLU as a light switch—only turning on (positive) features that matter, letting the network focus on relevant patterns while discarding noise. Without it, the "light" stays on for everything, blurring the picture.

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

- Pin memory: refers to allocating memory in a way that prevents the operating system from swapping it to disk (paging). Pinned memory is a limited system resource. Using too much can cause system instability.

- Prefetch Fector: controls the number of batches loaded in advance by each worker process to improve data loading efficiency. If GPU processing batch 1, worker will load batch 4 if prefetch is 4.

- Persistent Workers: keep DataLoader worker processes alive between epochs instead of recreating them. This eliminates creation of new processes every epoch and is memory efficient.

- Shuffles data every epoch if it is True.

Steps

1. Worker processes load 256 images from disk
2. Apply transforms to each image individually
3. Stack into batch tensor (256, 3, 32, 32)
4. Transfer to GPU

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

**.cudnn.benchmark**

- First few iterations, cuDNN measures performance for your specific setup. Caches the fastest algorithm found.

* Disable when input sizes are variable, or training runs are short.

**.cuda.matmul.allow_tf32**

- TF32 is different then FP32 since it is same range but reduced precision.

* Enable on newest GPU, training when speed > precision.

**.cudnn.allow_tf32**

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

**GradScaler**

Problems with FP16:

- Weight updates: with half precision, 1 + 0.0001 rounds to 1. autocast() takes care of this one.

- Vanishing gradients: with half precision, anything less than (roughly) 2e-14 rounds to 0, as opposed to single precision 2e-126. GradScaler() takes care of this one.

- Explosive loss: similar to the above, overflow is also much more likely with half precision. This is also managed by autocast() context.

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

51% CPU result with 20 epochs and gradient accumulation

prefetch factor and persistent_workers only works when num workers > 0

An asynchronous context manager in Python is an object that allows for the allocation and release of resources within asynchronous code, ensuring reliable setup and teardown logic even if the asynchronous operations encounter errors or interruptions.

RandAugment: better and stronger transforms from torchvision.

## Extra Notes:

if the number of hyperparameters is large you may prefer to use bigger validation splits. If the number of examples in the validation set is small (perhaps only a few hundred or so), it is safer to use cross-validation.

## Bayesian Optimization and Grid Search

## Graphing Results During Training

Use Matplotlib

- Validation loss per epoch and Training loss per epoch

Cutmix and mixup returns soft labels, making training accuracy calculate differently.

How to find a good initial learning rate?

**Log**

8/31/2025

```python
base_max_lr = 4e-3  # Higher for faster convergence in 60 epochs
batch_size = 256
scaling_factor = (batch_size / 256) ** 0.5  # Linear scaling
max_lr = base_max_lr * scaling_factor  # = 4e-3

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=max_lr,              # 4e-3 for 60 epochs
    epochs=num_epochs,          # 60
    steps_per_epoch=len(train_loader),
    pct_start=0.2,              # 20% warmup (12 epochs)
    anneal_strategy='cos',
    div_factor=25.0,            # Start LR = max_lr/25 = 1.6e-4
    final_div_factor=10000.0    # Final LR = max_lr/10000 = 4e-7
)

```

tried to run with 60 epochs but ended up with flattening validation error after epoch 30. Also added mixup and cutmix to transforms. Took about 40 minutes.

Learning rate scheduler:

Cosine annealing - have to set a higher initial learning rate compared to OneCycleLR, otherwise the optimal high learning rate never happens.

9/1:

Added a simpler transform, and updated batch size to 1024, with prefetch level 6.

Trained for 38 minutes and got 65%.

Trained for 31 minutes, got 63%

9/2

Added kaiming init to get down the initial loss to what it should be in theory

Reduce regularization in later epochs
Also increase the learning rate pct_start to 40%, and a slightly lower peak.

Trained for 29 min and got 63%
Trained for 39 min and got 65.58%. This is probably the capacity limit for 2 layer CNN.

Added a third block of CNN, added progressive augmentation reduction.

9/3

**Test Time Augmentation**:

**Finding Good Batch Size**

Smaller batch makes your gradient estimation more rough and less precise.

## Previously Tried Code

# scheduler = torch.optim.lr_scheduler.OneCycleLR(

# optimizer,

# max_lr=scaled_lr, # ~8e-3 (2x higher for larger batches)

# epochs=num_epochs,

# steps_per_epoch=len(train_loader), # Will be 44 instead of 176

# pct_start=0.25,

# anneal_strategy='cos',

# div_factor=20.0,

# final_div_factor=100.0

# )

# optimizer = torch.optim.AdamW(

# mlp.parameters(),

# lr=3e-3, # Increase from 1e-3 to 2e-3

# weight_decay=5e-4 # Reduce from 5e-3 to 1e-3

# )

# scheduler = torch.optim.lr_scheduler.OneCycleLR(

# optimizer,

# max_lr=4e-3, # Keep proven peak LR

# epochs=num_epochs, # 60

# steps_per_epoch=len(train_loader),

# pct_start=0.25, # 25% warmup (15 epochs) - longer peak

# anneal_strategy='cos',

# div_factor=20.0, # Start LR = 2e-4 (higher start)

# final_div_factor=200.0 # Final LR = 2e-5 (much higher final)

# )

# new_max_lr = 2e-3 \* (256 / 128)\*\*0.25

# scheduler = torch.optim.lr_scheduler.OneCycleLR(

# optimizer,

# max_lr=new_max_lr,

# epochs=num_epochs,

# steps_per_epoch=len(train_loader),

# pct_start=0.15, # Increase from 0.1 to 0.3

# anneal_strategy='cos'

# )

# base_max_lr = 4e-3 # Higher for faster convergence in 60 epochs

# batch_size = 256

# scaling_factor = (batch_size / 256) \*\* 0.5 # Linear scaling

# max_lr = base_max_lr \* scaling_factor # = 4e-3

# scheduler = torch.optim.lr_scheduler.OneCycleLR(

# optimizer,

# max_lr=max_lr, # 4e-3 for 60 epochs

# epochs=num_epochs, # 60

# steps_per_epoch=len(train_loader),

# pct_start=0.2, # 20% warmup (12 epochs)

# anneal_strategy='cos',

# div_factor=25.0, # Start LR = max_lr/25 = 1.6e-4

# final_div_factor=10000.0 # Final LR = max_lr/10000 = 4e-7

# )

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

# optimizer, T_max=num_epochs, eta_min=1e-5

# )

# scheduler = torch.optim.lr_scheduler.OneCycleLR(

# optimizer,

# max_lr=6e-3, # Reduce from 8e-3 to 6e-3 for stability

# epochs=num_epochs, # 50

# steps_per_epoch=len(train_loader),

# pct_start=0.35, # Longer warmup (17 epochs vs 12)

# anneal_strategy='cos',

# div_factor=15.0, # Higher start LR (4e-4 vs 4e-5)

# final_div_factor=300.0 # Much lower final LR (2e-5 vs 8e-5)

# )

Backend Notes:

psutil library

psutil.Process().memory_info().rss: gets current process, returns memory stats, and resident set size.

**X-Content-Type-Options: nosniff** -
Without this header, this could be dangerous:
User uploads "image.jpg" that's actually HTML with <script> tags
Browser might execute it instead of treating it as an image

**X-Frame-Options: Deny** - prevents clickjacking attacks. Prevents malicious sites from embedding API in hidden frames.

Without this header, malicious site could do:

<iframe src="your-api.com/predict" style="opacity:0;position:absolute">
User thinks they're clicking something else, but actually hits your API

**X-XSS-Protection: 1; mode=block**:

enable browser built in XSS filtering and blocks page if XSS found

Helps prevent scenarios like:
Malicious user submits: <script>steal_data()</script>
Browser detects and blocks instead of executing

**Strict-Transport-Security (HSTS)**:
Prevents downgrade attacks (forcing HTTP instead of HTTPS) and man in the middle attacks, ensures all requests use encrypted connections

max-age=31536000 = 1 year
includeSubDomains = applies to all subdomains too

**"default-src 'none'; frame-ancestors 'none'; base-uri 'none'"**

default-src 'none' - Blocks ALL resources by default (scripts, styles, images, etc.)
frame-ancestors 'none' - Prevents your API from being embedded in any iframe (redundant with X-Frame-Options but more modern)
base-uri 'none' - Prevents changing the document's base URL

**Referrer-Policy**

no referrer would control what info sent with requests:

- if user visits url, then clicks link to external site, the external site would see traces of original url. With no referrer, wouldn't see it

**Permissions-Policy** - microphone, camera, etc:

Disables browser API's that your API doesn't need.

**X-Permitted-Cross-Domain-Policies:**

- controls adobe flash and pdf cross domain access (pdf security, legacy stuff)

**Cache-control**

no-store - Don't save response anywhere (disk, memory)
no-cache - Always check with server before using cached version
must-revalidate - Force validation even for stale content
private - Only client can cache, not shared proxies

Sensitive data - Image classifications shouldn't be cached
Fresh predictions - Each request should get real-time results
Privacy - Prevents caching of user-uploaded images

"|" requires Python 3.10 or higher

# LSTM Intro --- New Section!

There is a cell state that runs linearly through an LSTM. Cell state contains gates which regulate adding or removal of info. Gates are sigmoids and mat muls that allow to flow (1) or not (0).

**Forget gate**: “What old info should I throw away?”

**Input gate**: “What new info should I add?”

**Cell state**: “What do I carry forward as memory?”

**Output gate**: “What should I say at this step?”

Step by Step Walkthrough

1. What information do we throw away? Looks at previous layer and outputs 0 for remove or 1 to keep.

Sigmoid(wei \* (prev_output, input) + bias)

2. What new info do we store in the cell state?

- sigmoid layer decides which values we update
- tanh layer creates vector of new candidate values that can be added

i = sigmoid(weig _ (prev_outpu, input) + bias)
C = tanh(wei _ (prev_output, input) + bias)

3. Actually do the update

C = old state _ Step 1 func + i _ C

Run through sigmoid decides what parts of state we want to output, through tanh to push between -1 and 1, then multiply output of sigmoid gate

o(t) = sigmoid(wei _ (prev_output, input) + bias)
h(t) = o(t) _ tanh(C)

First part of LSTM determines what percentage of long term memory it should remember

3 Blocks

1. % long term to remember

2. Potential long term memory

- multiply short term memory and input by their respective weights and add a bias before putting through tanh function

3. % potential memory to remember

- multiply short term memory and input by their respective weights and add a bias before putting through tanh function

- multiply the results from 2 and 3, and add this to the existing long term memory

4. Update short term memory

- Use new long term memory and use as input into tanh activation function.

- multiply short term memory and input by their respective weights and add a bias before putting through tanh function

- THIS IS THE OUTPUT GATE!

How much of this short term memory to pass on? We multiply the results of the two blocks from step 4.

LSTM reuses same weights and biases to handle data sequences of different lengths.

LSTM can process data sequentially even if data itself is not sequential.

# Step by Step for Simple LSTM Decoder

Linear layer to map the image features to match size of the LSTM.

Ex: let's say image feature tensor is (32, 2048) and our LSTM has hidden size of 512, then we need linear transform to convert (32, 2048) to (32, 512)

Word embedding layer creates an embedding layer that maps every single word indices in the vocab to a vector of size embed_dim

`torch.unsqueeze`: returns a new tensor with a dimension of size 1 inserted at specific position.

`torch.squeeze`: input of shape (A x 1 x B x 1 x C) is (A x B x C)

```python
        self.lstm = nn.LSTM(
            input_size = embed_dim,
            hidden_size = hidden_dim,
            num_layers = num_layers,
            dropout = 0.3,
            batch_first = True
        )
```

pytorch is initializing the W, U, b matrices for all 4 gates at once

# Attention

No notion of space, meaning that each vector has no concept of where it is.

Each example of batch dimension is working independently

Divide by sqrt head size - control variance at init since softmax would converge to 1 hot vectors if values are too big or too small.

# ML OPS Course

Biggest points for failure:

1. technically infeasible
2. never make leap to prod
3. no organization consensus
4.

# OAuth

In this case, the access token is my GitHub personal token. This is returned in the auth callback route.

Client ID: helps identify the app
Client Secret: basically a password. - should never be exposed.

After the user gives access, github will give a secret code. Then, a POST request will contain the secret, client id, and client secret.

.search in URLocation in react router gives back everything after the ? in the url

Implicit Grant: access token would be returned directly in the URL fragment.

PKCE: Would include PKCE specific parameters like code_verifier, code_challenge, etc.

Resources Owner Credentials Grant Type: the trusted client would get the resource owner's credential

Client Credentials Grant type: for microservices: using clientid and secret. This is for machine to machine communication, so it is very simple.

# Full Stack Deep Learning Course

## Development Infrastructure

Data parallelism

Model Parallelism: put each layer of model on GPU. You have to tuen amount of pipelining on batch size.

Tensor Parallelism: split up matrix operations on multiple GPU's.

## GPU's

- How much data fits into GPU?
- TFlops?
- Communication between CPU and GPU -> how fast is it?

## Memorization

Memorization Test: Only really gross issues with training will show up with this test.

Make sure to tune memorization tests to run quickly, so you can regularly run them.

The best option for testing training is to regularly run training with new data that's coming in from production. This is still expensive, but it is directly related to improvements in model development, not just testing for breakages. Setting this up requires a data flywheel similar to what we talked about in Lecture 1. Further tooling needed to achieve will be discussed down the line.

Models are functions themselves! Check if consistent values are being returned.

Watch out for shape errors, numerical errors, out of memory errors

Profile your code!

## Data Management

SQL, and Pandas

Airflow, Dagster, Prefect? For Data processing

Synthetic Data:

Data versioning:

Level 0 is bad: data just lives on file system
Level 1: snapshot data each time you train
Level 2: data versioned like code.
Level 3: specialized solutions for working with large data files

# Apply ML OPs to my workflow:

Data → Model Development → Training → Evaluation → Deployment → Monitoring → Iteration

DVC

| Challenge in ML                    | How DVC Helps                                                                  |
| ---------------------------------- | ------------------------------------------------------------------------------ |
| **Huge datasets/models**           | Uses efficient storage and caching so Git repos stay small.                    |
| **Reproducibility**                | Guarantees that a specific commit maps to a specific dataset and model hash.   |
| **Team collaboration**             | Multiple people can share and update data/models without emailing large files. |
| **Continuous training/deployment** | Works well in automated pipelines for retraining and redeployment.             |

## Testing:

Memorization tests can be used to easily check that the training loop is working correctly. If the model can't overfit a tiny dataset, something is wrong.

Small bugs are not going to show up, but big bugs will. If you include the length of run time in your coverage, and you see that the number of epochs to overfit is increasing, that is a sign that something is wrong.

Make the memorization tests easy to run, under 10 minutes is ideal.

- reduce dataset size
- turn off regularization
- reduce model size

Models are functions themselves! Check if consistent values are being returned.

Testing in production is the best way!

Track gradient norms. Infinity or NaN values are a sign of instability. Normalization is typically a cause of this. To fix, try Python 64 bit floats first!

## Deployment:

Batch prediction: Use a separate endpoint for batch predictions to improve throughput.

You can have reasonably fresh predictions to return to those users that are stored in your database.

Model as a Service: Use a microservice architecture to deploy your model as a standalone service. This allows for easier scaling and updating of the model without affecting other parts of your application.

## MLflow:

Logging artifacts:

- need job persistence for server restarts.
- task queue
  batch processing working pool
- caption personalization engine

- distributed caching, not needed for now

## Problems with current caching mechanism:

Not Distributed: Each worker process has its own separate cache

Worker 1 caches image A → stored only in Worker 1's memory
Worker 2 gets same image A → cache miss, recomputes
Lost on Restart: Server restarts = all cache gone

Memory Limits: Single process memory = your limit

No Eviction Policy: Dict grows forever until OOM

field(default_factory=list) - what does this mean
dataclasses in python

The error is happening because in Python dataclasses, all fields with default values must come after fields without defaults. You have created_at (no default) coming after results (has default).

Generator in typing

- A Generator is a special type of iterator that can yield multiple values over time, pausing its state between each yield. It allows you to iterate through a sequence of values without storing them all in memory at once.

job scheduling with threads

```text

┌─────────────────────────────────────────────────────────┐
│                    Client Request                        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  FastAPI Endpoint (/predict-batch)                       │
│  1. Create job in Redis (JobManager)                     │
│  2. Submit task to Queue                                 │
└────────────────────┬────────────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
┌──────────────────┐  ┌──────────────────────┐
│  Redis           │  │  Async Queue          │
│  (JobManager)    │  │  (AsyncJobQueue)      │
│                  │  │                       │
│  - Job Status    │  │  - Task Execution     │
│  - Progress      │  │  - Worker Pool        │
│  - Results       │  │  - Concurrency        │
│  - Persistence   │  │  - Priority           │
└──────────────────┘  └──────────┬────────────┘
                                 │
                      ┌──────────┴──────────┐
                      │                     │
                      ▼                     ▼
              ┌─────────────┐      ┌─────────────┐
              │  Worker 1   │      │  Worker 2   │
              │             │      │             │
              │ - Get task  │      │ - Get task  │
              │ - Process   │      │ - Process   │
              │ - Update    │      │ - Update    │
              │   Redis     │      │   Redis     │
              └─────────────┘      └─────────────┘
```

global keyword in Python:

- The global keyword in Python is used to declare that a variable inside a function refers to a global variable (a variable defined outside the function). This allows you to modify the global variable from within the function.

GIL in Python

- The Global Interpreter Lock (GIL) is a mutex that protects access to Python objects, preventing multiple native threads from executing Python bytecodes at once. This means that even in a multi-threaded Python program, only one thread can execute Python code at a time.

Implement Multi-Process captioning

Process 1: FastAPI Frontend (Main Process)
Handles all HTTP requests and WebSocket connections
Manages the job queue (your existing AsyncJobQueue)
Receives image upload requests
Sends jobs to the worker process
Returns results to clients
Manages Redis for job status tracking

Process 2: Model Worker (Separate Process)
Runs in complete isolation
Loads and keeps the Hugging Face caption model in memory
Receives job requests from the frontend
Processes images and generates captions
Sends results back to frontend
Can be restarted without affecting the main app

Use multiprocessing.Pipe to communicate between the FastAPI process and the model worker process.

Option 1: multiprocessing.Queue (Simplest)
Built into Python, no external dependencies
Use two queues: one for requests, one for responses
Frontend puts jobs in request queue
Worker takes from request queue, processes, puts results in response queue
Frontend monitors response queue

Implementation Steps

Step 1: Separate the Model Loading
Move all Hugging Face model code into a separate worker module
Keep the model initialization completely isolated
The worker process will load the model once on startup

Step 2: Define Message Protocol
Design the message format for communication:
Request message: Contains job_id, image bytes, parameters
Response message: Contains job_id, caption/error, status
Heartbeat message: Worker sends "I'm alive" signals
Use JSON or pickle for serialization

Step 3: Create Worker Process Manager
Write a function that starts the worker process
The worker runs an infinite loop: check queue → process job → send result → repeat
Handle graceful shutdown signals
Implement timeout detection (if no response in X seconds, assume worker crashed)

Step 4: Update FastAPI App
On startup, spawn the worker process
Monitor worker health with heartbeats
If worker dies, restart it automatically
On shutdown, gracefully terminate worker

Step 5: Modify Job Processing Flow
Old flow: FastAPI directly calls caption model

New flow:
FastAPI receives request
Create job in Redis (existing JobManager)
Send job to worker via queue/pipe
Worker receives job
Worker processes caption
Worker sends result back
FastAPI updates Redis with result
Client polls for status

Step 6: Handle Worker Failures
Timeout detection: If no response after 30 seconds, mark job as failed
Automatic restart: If worker process crashes, spawn a new one
Retry logic: Resubmit failed jobs (up to 3 times)
Circuit breaker: If worker keeps crashing, stop accepting new jobs temporarily

Step 7: Add Process Monitoring
Track worker process PID
Monitor queue depth (how many jobs waiting)
Log worker startup/shutdown
Expose metrics endpoint showing worker status

```text

FastAPI                    Worker Process
   │                            │
   │──── Send Job ────────────→ │  (Request Queue)
   │                            │
   │                       [Processing...]
   │                            │
   │←──── Get Result ───────────│  (Response Queue)
   │                            │

```

11/3/2025

- trying to refactor to separate processes. the generate_caption function is no longer being used. got to fix the caption worker process.
- trying to get rid of generate_caption function in websocket inference route.

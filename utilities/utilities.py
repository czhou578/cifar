import torch

# initial loss of a 100 class CIFAR

'''
Initial loss for 100 classes is 4.6052
'''
print(-torch.tensor(1/100).log())
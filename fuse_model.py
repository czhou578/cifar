
def fuse_model(model):
    """
    Fuses Conv-BN-ReLU layers in a model that uses nn.Sequential with OrderedDict.
    """
    modules_to_fuse = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Sequential):
            # Convert children to a list to allow indexing
            layer_list = list(module.children())
            # Get the string names of the layers
            layer_names = [n for n, _ in module.named_children()]

            for i in range(len(layer_list) - 2):
                if (isinstance(layer_list[i], nn.Conv2d) and
                    isinstance(layer_list[i + 1], nn.BatchNorm2d) and
                    isinstance(layer_list[i + 2], nn.ReLU)):

                    # Construct the full string names for fuse_modules
                    modules_to_fuse.append([
                        f'{name}.{layer_names[i]}',
                        f'{name}.{layer_names[i+1]}',
                        f'{name}.{layer_names[i+2]}'
                    ])

    if modules_to_fuse:
        print(f"Fusing {len(modules_to_fuse)} layers...")
        # Fusion must be done in eval mode.
        model.eval()
        torch.quantization.fuse_modules(model, modules_to_fuse, inplace=True)
    return model
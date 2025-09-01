def prune_and_finetune_model(model, amount=0.3, finetune_epochs=5):
    """
    Applies structured pruning to the model and finetunes it to recover lost progress
    """

    print(f"Pruning {amount*100}% of channels from all Conv2d layers...")

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)

    optimizer.param_groups[0]['lr'] = 1e-5

    for epoch in range(finetune_epochs):
        model.train()
        for data in train_loader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)

            with autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = loss_function(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        print(f"Fine-tuning epoch {epoch+1}/{finetune_epochs} complete.")

    print("\nMaking pruning permanent...")
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.remove(module, 'weight')

    return model
import pytest
import torch
from model import MLP

class TestModel:
    def test_model_memorization(self):
        """Test model can memorize small dataset"""
        model = MLP()
        # Create tiny dataset of 10 samples
        tiny_data = torch.randn(10, 3, 32, 32)
        tiny_labels = torch.randint(0, 100, (10,))
        
        # Train until perfect accuracy
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(100):
            outputs = model(tiny_data)
            loss = torch.nn.CrossEntropyLoss()(outputs, tiny_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Should achieve near-perfect accuracy
        with torch.no_grad():
            outputs = model(tiny_data)
            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == tiny_labels).float().mean()
            assert accuracy > 0.95, f"Memorization test failed: {accuracy}"
    
    def test_model_consistency(self):
        """Test model returns consistent outputs"""
        model = MLP()
        model.eval()
        
        input_tensor = torch.randn(1, 3, 32, 32)
        
        with torch.no_grad():
            output1 = model(input_tensor)
            output2 = model(input_tensor)
            
        assert torch.allclose(output1, output2, atol=1e-6), "Model outputs not consistent"
    
    def test_model_shapes(self):
        """Test model input/output shapes"""
        model = MLP()
        batch_sizes = [1, 16, 32, 64]
        
        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, 3, 32, 32)
            output = model(input_tensor)
            
            assert output.shape == (batch_size, 100), f"Wrong output shape: {output.shape}"
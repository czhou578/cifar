# This is a helper script to run separately.
# It does not go into your main model.py file.
import torch
from model import MLP # Assuming your model class is in model.py

def find_max_batch_size(model, device, low=1, high=2048):
    """
    Performs a binary search to find the maximum batch size that fits in memory.
    """
    max_size = low
    while low <= high:
        mid = (low + high) // 2
        if mid == 0: # Avoid batch size of 0
            low = mid + 1
            continue
            
        print(f"Testing batch size: {mid}...")
        try:
            # Create dummy data for the test
            dummy_input = torch.randn(mid, 3, 32, 32, device=device)
            dummy_target = torch.randint(0, 100, (mid,), device=device)
            
            # Perform a single forward and backward pass
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            output = model(dummy_input)
            loss = torch.nn.functional.cross_entropy(output, dummy_target)
            loss.backward()
            optimizer.zero_grad(set_to_none=True)
            
            # If successful, this batch size is a candidate
            max_size = mid
            print(f"  -> Success. Trying larger sizes.")
            low = mid + 1 # Try a larger size
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  -> Failed (Out of Memory). Trying smaller sizes.")
                high = mid - 1 # Try a smaller size
                # Clear cache after OOM
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                raise e # Re-raise other errors

    print(f"\nMaximum batch size found: {max_size}")
    return max_size

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP().to(device)
    model.train()
    
    find_max_batch_size(model, device)
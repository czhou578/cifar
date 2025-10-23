import hashlib
import torch
from typing import List, Dict, Any, Optional

'''
This is an implementation of a simple in-memory cache for storing image predictions.
For a production system, consider using a more robust caching solution like Redis or Memcached.
'''

class PredictionCache:
    def __init__(self, size):
        self.cache = {}
        self.max_size = size
        self.priority = []
    
    def _get_image_hash(self, image_tensor: torch.Tensor) -> str:
        """Generate hash for image tensor"""

        return hashlib.md5(image_tensor.cpu().numpy().tobytes()).hexdigest()

    def get(self, image_tensor: torch.Tensor) -> Optional[List[Dict[str, Any]]]:
        """Get cached prediction"""

        image_hash = self._get_image_hash(image_tensor)

        if image_hash in self.cache:
            prior_idx_key = self.priority.index(image_hash)
            self.priority.insert(0, self.priority.pop(prior_idx_key))
            return self.cache[image_hash]
        else:
            return None
        
    def set(self, image_tensor: torch.Tensor, predictions: List[Dict[str, Any]]):
        """Cache prediction"""

        image_hash = self._get_image_hash(image_tensor)

        self.cache[image_hash] = predictions

        if image_hash not in self.priority:
            self.priority.insert(0, image_hash)
        
        else:
            prior_idx_key = self.priority.index(image_hash)
            self.priority.insert(0, self.priority.pop(prior_idx_key))
        
        if len(self.cache.keys()) > self.max_size:
            key = self.priority.pop()
            del key

prediction_cache = PredictionCache(size=400)

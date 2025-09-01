import torch
import torch.nn.functional as F

def multimodal_contrastive_loss(image_embeddings, text_embeddings, temperature=0.07):
    """
    Computes multimodal contrastive loss between image and text embeddings.
    
    Args:
        image_embeddings (torch.Tensor): Image embeddings of shape (batch_size, embedding_dim).
        text_embeddings (torch.Tensor): Text embeddings of shape (batch_size, embedding_dim).
        temperature (float): Temperature parameter for scaling logits.
    
    Returns:
        torch.Tensor: Contrastive loss value.
    """

    
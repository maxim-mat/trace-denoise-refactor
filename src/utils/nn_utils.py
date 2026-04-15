import torch

def _downsample_mask(mask, factor=2):
    """Downsample a boolean mask (B, L) by max-pooling.
    
    If any position in a pooling window is True (real), the output is True.
    Uses float max-pool then converts back to bool.
    """
    if mask is None:
        return None
    # (B, L) -> (B, 1, L) for max_pool1d
    m = mask.float().unsqueeze(1)
    m = torch.nn.functional.max_pool1d(m, kernel_size=factor)
    return m.squeeze(1).bool()  # (B, L//factor)

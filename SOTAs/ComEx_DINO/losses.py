import torch

def pearson_correlation(x, y):
    """
    Compute the Pearson correlation coefficient between each pair of features in x and y.
    
    Parameters:
        x (torch.Tensor): A tensor of shape (batch_size, dim_feature).
        y (torch.Tensor): A tensor of shape (batch_size, dim_feature).
        
    Returns:
        torch.Tensor: A tensor of shape (batch_size,) containing the Pearson correlation coefficients.
    """
    # Compute means
    mean_x = torch.mean(x, dim=1, keepdim=True)
    mean_y = torch.mean(y, dim=1, keepdim=True)
    # Compute numerator
    numerator = torch.sum((x - mean_x) * (y - mean_y), dim=1)
    # Compute denominators
    denominator_x = torch.sqrt(torch.sum((x - mean_x) ** 2, dim=1))
    denominator_y = torch.sqrt(torch.sum((y - mean_y) ** 2, dim=1))
    # Compute Pearson correlation coefficient
    correlation = numerator / (denominator_x * denominator_y)
    
    return correlation
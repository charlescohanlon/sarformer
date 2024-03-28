import torch
import torch.nn as nn


class SARFormer(nn.Module):
    # Create random tensors to represent latent feature representations from swin_v2 and ALBERT
    # Note that dim1 is the batch size, dim2 is the sequence length (for ALBERT) and
    # window_size^2 for swin_v2, and 768 is the hidden dimension size for both models.
    tensor_swin = torch.randn(64, 100, 768)  # assuming batch size 64, window size 10
    tensor_albert = torch.randn(
        64, 200, 768
    )  # assuming batch size 64, sequence length 200

    # Concatenate along the second dimension (dim=1)
    concatenated_tensor = torch.cat((tensor_swin, tensor_albert), dim=1)

    print(concatenated_tensor.shape)  # Output: torch.Size([64, 300, 768])

    # We would then input the concatenated tensor into the decoder

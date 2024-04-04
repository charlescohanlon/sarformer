import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationDecoder(nn.Module):
    """
    Segmentation Decoder for text and image concatenated input.
    
    Args:
        text_latent_size (int): Size of text latent space.
        image_latent_size (int): Size of image latent space.
        output_size (int): Output size for segmentation.
        hidden_sizes (list of int): Hidden layer sizes.
        activation (nn.Module, optional): Activation function for the final layer.
        upsampling_factor (int, optional): Upsampling factor for the final output.
    """
    def __init__(self, text_latent_size, image_latent_size, output_size, hidden_sizes,
                 activation=None, upsampling_factor=2):
        super(SegmentationDecoder, self).__init__()

        self.text_latent_size = text_latent_size
        self.image_latent_size = image_latent_size

        input_size = text_latent_size + image_latent_size

        # Decoder layers
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        self.decoder = nn.Sequential(*layers)

        self.upsampling_factor = upsampling_factor

        if activation is not None:
            self.activation = activation
        else:
            self.activation = nn.Identity()

    def forward(self, text_latent, image_latent):
        # Combine text and image latent features
        combined_latent = torch.cat((text_latent, image_latent), dim=1)

        output = self.decoder(combined_latent)

        output = self.activation(output)

        return output

text_latent_size = 256  
image_latent_size = 512 
output_size = 10
hidden_sizes = [768, 768] 

decoder = SegmentationDecoder(text_latent_size, image_latent_size, output_size, hidden_sizes, activation=nn.Softmax(dim=1))

text_latent = torch.randn(1, text_latent_size)  # Example text latent vector
image_latent = torch.randn(1, image_latent_size)  # Example image latent vector

output = decoder(text_latent, image_latent)
print("Segmentation output shape:", output.shape)
print(output)
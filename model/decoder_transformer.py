import torch
import torch.nn as nn
from torch.nn import functional as F


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_dim=768,
        num_decoder_layers=12,
        num_attention_heads=12,
        window_size=8,
        num_modalities=3,
        dropout=0.1,
        output_size=512,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_decoder_layers = num_decoder_layers
        self.num_attention_heads = num_attention_heads
        self.window_size = window_size
        self.num_modalities = num_modalities
        self.dropout = dropout
        self.output_size = output_size

        # Modality-specific embeddings
        # self.modality_embeddings = nn.ModuleList([
        #     nn.Embedding(hidden_dim, hidden_dim) for _ in range(num_modalities)
        # ])

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_attention_heads
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=num_decoder_layers
        )

        # Output linear layer
        self.output_linear = nn.Linear(hidden_dim, output_size)

        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: dict):
        # Apply modality-specific embeddings
        # modality_embedded_inputs = [
        #     embedding(concatenated_input[:, :, i].long()) for i, embedding in enumerate(self.modality_embeddings)
        # ]
        # concatenated_input = torch.stack(modality_embedded_inputs, dim=2)

        # TODO: check if "masked" in x which is a dictionary
        # Transformer decoder
        transformer_output = self.transformer_decoder(
            tgt=concatenated_input,
            memory=concatenated_input,
            tgt_mask=sequence_mask,
            memory_mask=sequence_mask,
        )

        # Apply layer normalization and dropout
        transformer_output = self.layer_norm(transformer_output)
        transformer_output = self.dropout_layer(transformer_output)

        # Output linear layer
        probability_map = self.output_linear(transformer_output)

        return probability_map


batch_size = 64
hidden_dim = 768
text_rep = torch.randn(batch_size, 512, hidden_dim)
image_rep = torch.randn(batch_size, 2000, hidden_dim)
tabular_rep = torch.randn(batch_size, 256, hidden_dim)

# Concatenate representations along the last dimension
concatenated_input = torch.cat((text_rep, image_rep, tabular_rep), dim=1)

# print(concatenated_input.shape)

# Initialize Decoder
decoder = Decoder(
    hidden_dim=768,
    num_decoder_layers=12,
    num_attention_heads=12,
    window_size=8,
    num_modalities=3,
    dropout=0.1,
    output_size=512,
)

output = decoder(concatenated_input)
print(output.shape)

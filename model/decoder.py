import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Decoder(nn.Module):
    def __init__(
        self, hidden_dim, num_decoder_layers, num_attention_heads, window_size
    ):
        # Many more parameters to be added later!
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_decoder_layers = num_decoder_layers
        self.num_attention_heads = num_attention_heads
        self.window_size = window_size

        # Shared positional embeddings (not sure how/if this will be implemented)
        self.positional_embedding = nn.Parameter(
            torch.randn(window_size * window_size, hidden_dim)
        )

        # Shared modality embeddings (also not sure about this with only 3 modalities)
        self.modality_embedding = nn.Parameter(torch.randn(3, hidden_dim))

        # Transformer decoder (based on "Attention is All You Need" paper)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=hidden_dim, nhead=num_attention_heads
            ),
            num_layers=num_decoder_layers,
        )

        # Shared output embedding (final linear layer)
        self.output_embedding = nn.Linear(hidden_dim, 1)

    def forward(self, concatenated_input, sequence_mask=None):
        # Add modality embeddings (just adding for now, will likely change later)
        concatenated_input += self.modality_embedding

        # Add positional embeddings
        concatenated_input += self.positional_embedding

        # Apply sequence mask if given as parameter
        if sequence_mask is not None:

            # Transformer decoder output
            transformer_output = self.transformer_decoder(
                tgt=concatenated_input,
                memory=concatenated_input,
                tgt_mask=sequence_mask,
                memory_mask=sequence_mask,
            )

            # Remove batch dimension
            transformer_output = transformer_output.squeeze(0)

            # Apply output embedding to get probability map
            probability_map = self.output_embedding(transformer_output)

            return probability_map

        else:
            pass

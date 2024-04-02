import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

class Decoder(nn.Module):
    def __init__(
        self, hidden_dim = 768, num_decoder_layers = 12, num_attention_heads = 12, window_size = 8,
        num_modalities=3, dropout=0.1
    ):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_decoder_layers = num_decoder_layers
        self.num_attention_heads = num_attention_heads
        self.window_size = window_size
        self.num_modalities = num_modalities
        self.dropout = dropout

        # Modality-specific embeddings (just for show)
        self.modality_embeddings = nn.ModuleList([
            nn.Embedding(hidden_dim, hidden_dim) for _ in range(num_modalities)
        ])

        # Positional encoding (just for show)
        self.positional_encoding = nn.Parameter(
            torch.randn(window_size * window_size, hidden_dim)
        )

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_attention_heads
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=num_decoder_layers
        )

        # Output linear layer
        self.output_linear = nn.Linear(hidden_dim, 1)

        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, concatenated_input, sequence_mask=None):
        # Modality-specific embeddings
        modality_embedded_inputs = [
            embedded(concatenated_input[:, i]) for i, embedded in enumerate(self.modality_embeddings)
        ]
        concatenated_input = torch.stack(modality_embedded_inputs, dim=1)

        # Positional encoding
        concatenated_input += self.positional_encoding

        # Transformer decoder
        transformer_output = self.transformer_decoder(
            tgt=concatenated_input,
            memory=concatenated_input,
            tgt_mask=sequence_mask,
            memory_mask=sequence_mask
        )

        # Apply layer normalization and dropout
        transformer_output = self.layer_norm(transformer_output)
        transformer_output = self.dropout_layer(transformer_output)

        # Output linear layer
        probability_map = self.output_linear(transformer_output)

        return probability_map.squeeze(2)

import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, input_dim = 539, num_heads = 12, num_layers = 1, hidden_dim = 804, output_dim = 512):
        super().__init__()
        self.hidden_dim = hidden_dim
        # self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim,
                                                                     nhead=num_heads,
                                                                     dim_feedforward=hidden_dim * 4)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=num_layers)
        
        self.output_proj1 = nn.Linear(hidden_dim, output_dim)  # Project to the output dimension
        self.output_proj2 = nn.Linear(input_dim, output_dim)  # Project to the output dimension

    def forward(self, input_dim, inputs):
        embed, masked = inputs["embed"], inputs["masked"]
        
        embed = nn.Linear(embed.size(2), self.hidden_dim)(embed)
        
        # Using embed for target and masked for memory for cross-attention in the decoder
        tgt = embed.permute(1, 0, 2)  # Sequence length, Batch size, Embedding dim
        memory = masked.permute(1, 0, 2)
        
        # might need positional encodings here
        output = self.transformer_decoder(tgt, memory)
        # may need to add in tgt_mask=sequence_mask and memory_mask=sequence_mask, but this
        # was causing me problems so I removed it for now
        output = self.output_proj1(output.permute(1, 0, 2))
        output = self.output_proj2(output.permute(0, 2, 1))
        # Somehow need to convert to probability map
        return output
import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, num_heads=1, num_layers=1, hidden_dim=2561, d_model=514):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.d_model = d_model
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=d_model,
                                                                     nhead=num_heads,
                                                                     dim_feedforward=hidden_dim)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=num_layers)
        self.input_proj_embed = nn.Linear(d_model, d_model)
        self.input_proj_masked = nn.Linear(hidden_dim, d_model)

    def forward(self, inputs):
        embed, masked = inputs["embed"], inputs["masked"]
        #embed = embed.permute(0, 2, 1)
        #masked = masked.permute(0, 2, 1)
        tgt = self.input_proj_embed(embed)  # Project embed to d_model dimension
        memory = self.input_proj_masked(masked)  # Project masked to d_model dimension
        
        print(tgt.size())  # Expected size: (batch_size, seq_len, d_model)
        print(memory.size())  # Expected size: (batch_size, seq_len, d_model)
        
        output = self.transformer_decoder(tgt, memory)
        
        print(output.size())  # Expected output size: (seq_len, batch_size, d_model)
        return output

# Example usage:

# decoder = Decoder()
# inputs = {
#     "embed": torch.randn(768, 1, 91),  # (seq_len, batch_size, embed_dim)
#     "masked": torch.randn(768, 1, 2075)    # (seq_len, batch_size, masked_dim)
# }
# output = decoder(inputs)
# print(output)
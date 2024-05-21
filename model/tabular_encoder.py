import torch
import torch.nn as nn
import torch.nn.functional as F

class TabularEncoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_layers=4,
        layer_dim=None,
        act_layer=nn.ReLU,
        dropout_prob=0.0,
        use_batch_norm=True,
        mask_proportion=0,
        reshape_dim=None
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        self.use_batch_norm = use_batch_norm
        self.masked_proportion = mask_proportion

        # If layer_dim is not provided, linearly increase embedding dim over layers
        if layer_dim is None:
            layer_dim = [
                int(in_dim + i * ((out_dim - in_dim) / num_layers))
                for i in range(num_layers + 1)
            ]

        # Add dropout layer
        self.dropout = nn.Dropout(p=dropout_prob)

        # Add layers with activation and batch normalization
        for i in range(num_layers):
            self.layers.append(
                nn.Linear(
                    in_features=layer_dim[i],
                    out_features=layer_dim[i + 1],
                )
            )
            if self.use_batch_norm:
                self.layers.append(nn.BatchNorm1d(layer_dim[i + 1]))

            self.layers.append(act_layer())

        self.reshape_dim = reshape_dim

    def forward(self, x):
        #self.eval()
        mask = None
        if self.masked_proportion is not None:
            num_columns = x.shape[1]
            num_masked_columns = int(self.masked_proportion * num_columns)
            mask = torch.ones_like(x)
            mask[:, torch.randperm(num_columns)[:num_masked_columns]] = 0

        for layer in self.layers:
            x = layer(x)

        x = self.dropout(x)
        
        if self.reshape_dim is not None:
            batch_size = x.size(0)
            something = x.size(1)
            hidden_dim = x.size(2)
            x = x.view(batch_size, something, hidden_dim)
        
        if self.masked_proportion is not None:
            return mask.unsqueeze(1), x.unsqueeze(1)
        return x


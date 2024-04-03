import torch
import torch.nn as nn
import torch.nn.functional as F

class TabularEncoder(nn.Module):
    def __init__(
        self, in_dim, out_dim, num_layers=4, layer_dim=None,
        act_layer=nn.ReLU, dropout_prob=0.0, use_batch_norm=True
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        self.use_batch_norm = use_batch_norm

        # If layer_dim is not provided, linearly increase embedding dim over layers
        if layer_dim is None:
            layer_dim = [(in_dim + i * ((out_dim - in_dim) // num_layers)) for i in range(1, num_layers)]

        # Add dropout layer
        self.dropout = nn.Dropout(p=dropout_prob)

        # Add layers with activation and batch normalization
        for i in range(num_layers):
            self.layers.append(nn.Linear(in_features=in_dim if i == 0 else layer_dim[i - 1], out_features=layer_dim[i]))
            if self.use_batch_norm:
                self.layers.append(nn.BatchNorm1d(layer_dim[i])) # Hopefully using population statistics
            self.layers.append(act_layer()) # may want to not include on final layer

        # Add output layer
        self.layers.append(nn.Linear(in_features=layer_dim[-1], out_features=out_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.dropout(x)
        return x

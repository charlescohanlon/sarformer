import torch
import torch.nn as nn
import torch.nn.functional as F

class TabularEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, act_layer=nn.ReLU, dropout_prob=0.0, use_batch_norm=True):
        super().__init__()

        self.layers = nn.ModuleList()
        self.use_batch_norm = use_batch_norm

        # Linearly increases embedding dim from in_dim to out_dim over 4 FFNs
        l1_dim = in_dim + 1 * ((out_dim - in_dim) // 4)
        l2_dim = in_dim + 2 * ((out_dim - in_dim) // 4)
        l3_dim = in_dim + 3 * ((out_dim - in_dim) // 4)

        # Add dropout layer
        self.dropout = nn.Dropout(p=dropout_prob)

        # Add layers with activation and batch normalization
        self.layers.append(nn.Linear(in_features=in_dim, out_features=l1_dim))
        if self.use_batch_norm:
            self.layers.append(nn.BatchNorm1d(l1_dim)) # Hopefully using population statistics
        self.layers.append(act_layer())

        self.layers.append(nn.Linear(in_features=l1_dim, out_features=l2_dim))
        if self.use_batch_norm:
            self.layers.append(nn.BatchNorm1d(l2_dim))
        self.layers.append(act_layer())

        self.layers.append(nn.Linear(in_features=l2_dim, out_features=l3_dim))
        if self.use_batch_norm:
            self.layers.append(nn.BatchNorm1d(l3_dim))
        self.layers.append(act_layer())

        self.layers.append(nn.Linear(in_features=l3_dim, out_features=out_dim))
        if self.use_batch_norm:
            self.layers.append(nn.BatchNorm1d(out_dim))
        self.layers.append(act_layer())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.dropout(x)
        return x

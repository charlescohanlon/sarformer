import torch.nn as nn


class TabularEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, act_layer):
        super().__init__()

        # Linearly increases embedding dim from in_dim to out_dim over 4 FFNs
        l1_dim = in_dim + 1 * ((out_dim - in_dim) // 4)
        l2_dim = in_dim + 2 * ((out_dim - in_dim) // 4)
        l3_dim = in_dim + 3 * ((out_dim - in_dim) // 4)

        # NOTE: may want to add drop out or something

        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=l1_dim),
            act_layer(),
            nn.Linear(in_features=l1_dim, out_features=l2_dim),
            act_layer(),
            nn.Linear(in_features=l2_dim, out_features=l3_dim),
            act_layer(),
            nn.Linear(in_features=l3_dim, out_features=out_dim),
            act_layer(),
        )

    def forward(self, x):
        return self.mlp(x)

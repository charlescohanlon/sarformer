import torch
import torch.nn as nn
import torch.nn.functional as F


class TabularEncoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_layers=4,
        layer_dims=None,
        act_layer=nn.ReLU,
        dropout_prob=0.0,
        norm_layer=nn.BatchNorm1d,
        mask_proportion=0,
        mask_token=0,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        self.norm = norm_layer
        self.mask_proportion = mask_proportion
        self.mask_token = mask_token

        if mask_proportion:
            self.num_masked = round(in_dim * mask_proportion)

        # linearly increase embedding dim over each layers
        if layer_dims is None:
            layer_dims = [
                int(in_dim + i * ((out_dim - in_dim) / num_layers))
                for i in range(num_layers + 1)
            ]

        self.dropout = nn.Dropout(p=dropout_prob)

        for i in range(num_layers):
            self.layers.append(
                nn.Linear(
                    in_features=layer_dims[i],
                    out_features=layer_dims[i + 1],
                )
            )

            if norm_layer:
                self.layers.append(norm_layer(layer_dims[i + 1]))

            self.layers.append(act_layer())

    def forward(self, x):
        if self.mask_proportion:
            masked_full_x, mask, x = self.mask(x)

        for layer in self.layers:
            x = layer(x)

        x = self.dropout(x)

        if self.mask_proportion:
            return masked_full_x, mask, x

        return x

    def mask(self, x):
        with torch.no_grad():
            B, num_features = x.shape

            # choose random mask indices
            chosen_idxs = (
                torch.randint(high=num_features, size=(B, num_features))
                .argsort(dim=1)[:, : self.num_masked]
                .flatten()
            )

            # create corresponding batch indices
            batch_idxs = torch.arange(B).repeat_interleave(self.num_masked)

            mask = torch.ones_like(x)
            mask[batch_idxs, chosen_idxs] = 0

            masked_full_x = x.detach().clone()
            masked_full_x[~mask.to(torch.bool)] = self.mask_token

            # NOTE: the we're not shrinking the vector here
            # doing so wouldn't make sense in the context of a MLP
            x *= mask

            return masked_full_x, mask, x

import torch
from torch import nn
import torch.nn.functional as F
from transformers import RobertaConfig, RobertaModel


class RoBERTa(nn.Module):
    def __init__(
        self,
        max_length=512,
        pad_token=1,  # RoBERTa tokenizer pad token
        mask_proportion=0.0,
        mask_token=0,
    ):
        super().__init__()
        self.max_length = max_length
        self.pad_token = pad_token
        self.mask_proportion = mask_proportion
        self.mask_token = mask_token

        config = RobertaConfig()
        self.model = RobertaModel(config, add_pooling_layer=False)

    def forward(self, x):
        seq_len = x.shape[1]
        # a mask is required even when mask_proportion=0 b/c RobertaModel wants
        # padding to be masked
        mask = torch.ones_like(x)
        mask = F.pad(mask, (0, self.max_length - seq_len))

        # pad to full length
        x = F.pad(x, (0, self.max_length - seq_len), value=self.pad_token)

        if self.mask_proportion:
            masked_full_x, mask = self.create_mask(x, mask, seq_len)

        # mask applied in HF model class
        x = self.model(x, attention_mask=mask).last_hidden_state

        if self.mask_proportion:
            return masked_full_x, mask, x

        return x

    def create_mask(self, x, mask, seq_len):
        B = x.shape[0]
        num_masked = round(seq_len * self.mask_proportion)

        # choose random mask indices
        chosen_idxs = (
            torch.randint(high=seq_len, size=(B, seq_len))
            .argsort(dim=1)[:, :num_masked]
            .flatten()
        )

        # create corresponding batch indices
        batch_idxs = torch.arange(B).repeat_interleave(num_masked)

        mask[batch_idxs, chosen_idxs] = 0

        masked_full_x = x.detach().clone()
        masked_full_x[~mask.to(torch.bool)] = self.mask_token

        return masked_full_x, mask

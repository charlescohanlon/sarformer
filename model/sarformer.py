import torch
import torch.nn as nn
from swinv2_encoder import SwinTransformerV2
from bert_encoder import BertEncoder
from tabular_encoder import TabularEncoder
from decoder import Decoder


class SARFormer(nn.Module):
    def __init__(self, img_size, dim_embed, num_tab_features, device):
        super().__init__()
        torch.set_default_device(device)
        self.swin_encoder = SwinTransformerV2(
            img_size=img_size, patch_size=4, in_chans=4
        )

        # Parameters taken from Google's BERT implementation at https://arxiv.org/pdf/1810.04805.pdf
        self.bert_encoder = BertEncoder(
            d_model=dim_embed,
            nhead=12,
            num_encoder_layers=12,
            dim_feedforward=3072,
            activation="gelu",
        )

        self.tabular_encoder = TabularEncoder(
            num_tab_features,
            dim_embed,
            act_layer=nn.ReLU,
        )

        self.decoder = Decoder()

    def forward(self, image_tensor, text_tensor, tabular_tensor):
        swin_embed = self.swin_encoder(image_tensor)
        bert_embed = self.bert_encoder(text_tensor)
        tab_embed = self.tabular_encoder(tabular_tensor)

        # Concat along sequence dimension
        cat_embed = torch.cat((swin_embed, bert_embed, tab_embed), dim=1)

        return self.decoder(cat_embed)

import torch
import torch.nn as nn
from swinv2_encoder import SwinTransformerV2
from bert_encoder import BertEncoder
from tabular_encoder import TabularEncoder
from model.decoder_transformer import Decoder


class SARFormer(nn.Module):
    def __init__(
        self, img_size, dim_embed, num_tab_features, mask_proportions={}, device="cuda"
    ):
        super().__init__()
        torch.set_default_device(device)
        self.mask_proportions = mask_proportions
        self.swin_encoder = SwinTransformerV2(
            img_size=img_size,
            patch_size=4,
            in_chans=4,
            mask_proportion=self.mask_proportions["swin"],
        )

        self.bert_encoder = BertEncoder()

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

        # if mask proportions specified, forward() returned the masked input with the embeddings
        if self.mask_proportions:
            swin_masked, swin_embed = swin_embed
            bert_masked, bert_embed = bert_embed
            tab_masked, tab_embed = tab_embed

        # Concat along sequence dimension
        cat_embed = torch.cat((tab_embed, swin_embed, bert_embed), dim=1)
        if self.mask_proportions:
            cat_masked = torch.cat((tab_masked, swin_masked, bert_masked), dim=1)

            # "embed" is the shrunken embeddings (input to the cross-attention sublayer)
            # "masked" is the input with the mask applied (input to the mhsa sublayer)
            return self.decoder({"embed": cat_embed, "masked": cat_masked})

        return self.decoder({"embed": cat_embed})

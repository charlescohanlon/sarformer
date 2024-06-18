import torch
import torch.nn as nn
from model.swinv2_encoder import SwinTransformerV2
from model.bert_encoder import BertEncoder
from model.tabular_encoder import TabularEncoder
from model.transformer_decoder import Decoder
import torch.nn.functional as F


class SARFormer(nn.Module):
    def __init__(
        self,
        img_size=512,
        dim_embed=768,
        num_tab_features=10,
        text_max_seq_len=1024,
        mask_proportions={},
    ):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(device)
        self.mask_proportions = mask_proportions

        self.swin_encoder = SwinTransformerV2(
            img_size=img_size,
            patch_size=4,
            in_chans=4,
            mask_proportion=self.mask_proportions["swin"],
        )

        self.bert_encoder = BertEncoder(
            embedding_layers=12,
            max_tokens=text_max_seq_len,
            mask_proportion=self.mask_proportions["bert"],
        )

        self.tabular_encoder = TabularEncoder(
            num_tab_features,
            dim_embed,
            act_layer=nn.ReLU,
            mask_proportion=self.mask_proportions["tabular"],
            use_batch_norm=False,
        )

        self.decoder = Decoder()

    def forward(self, image_tensor, text_tensor, tabular_tensor):
        bert_embed = self.bert_encoder(text_tensor)
        swin_embed = self.swin_encoder(image_tensor)
        tab_embed = self.tabular_encoder(tabular_tensor)

        # if mask proportions specified, forward() returned the masked input with the embeddings
        if self.mask_proportions:
            swin_masked, swin_embed = swin_embed
            bert_masked, bert_embed = bert_embed
            tab_masked, tab_embed = tab_embed

        # concat along sequence dimension
        swin_embed = torch.tensor(swin_embed).transpose(2, 1)
        cat_embed = torch.cat((tab_embed, swin_embed, bert_embed), dim=1)
        if self.mask_proportions:
            swin_masked = F.pad(swin_masked.reshape(32, 2048, 512), (0, 256))
            tab_masked = F.pad(tab_masked, (0, 768 - tab_masked.size(2)))
            cat_masked = torch.cat((tab_masked, swin_masked, bert_masked), dim=1)

            # "embed" is the shrunken embeddings (input to the cross-attention sublayer)
            # "masked" is the input with the mask applied (input to the mhsa sublayer)
            print(cat_embed.shape)
            print(cat_masked.shape)
            return self.decoder(inputs={"embed": cat_embed, "masked": cat_masked})

        return self.decoder({"embed": cat_embed})

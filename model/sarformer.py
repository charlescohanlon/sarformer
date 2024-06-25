import torch
import torch.nn as nn
from model.swinv2_encoder import SwinTransformerV2
from model.roberta_encoder import RoBERTa
from model.tabular_encoder import TabularEncoder
from model.transformer_decoder import Decoder
import torch.nn.functional as F


class SARFormer(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        img_size=512,
        img_chans=4,
        swin_patch_size=8,
        swin_depths=[2, 2, 18, 2],
        swin_window_size=6,
        swin_pretrained_window_sizes=[0, 0, 0, 0],
        swin_use_ape=False,
        text_max_tokens=512,
        roberta_num_hidden_layers=12,
        roberta_ape=True,
        tabular_num_features=19,
        tabular_num_hidden_layers=4,
        tabular_layer_dims=None,
        tabular_dropout_prob=0,
        decoder_num_heads=1,
        decoder_num_blocks=1,
        mask_proportions={},
        mask_token=0,
    ):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(device)
        self.mask_proportions = mask_proportions

        self.swin_encoder = SwinTransformerV2(
            img_size=img_size,
            patch_size=swin_patch_size,
            in_chans=img_chans,
            embed_dim=embed_dim // 8,
            depths=swin_depths,
            window_size=swin_window_size,
            ape=swin_use_ape,
            pretrained_window_sizes=swin_pretrained_window_sizes,
            mask_proportion=mask_proportions["swin"],
            mask_token=mask_token,
        )

        self.roberta_encoder = RoBERTa(
            embed_dim=embed_dim,
            num_hidden_layers=roberta_num_hidden_layers,
            max_num_tokens=text_max_tokens,
            mask_proportion=mask_proportions["roberta"],
            mask_token=mask_token,
            ape=roberta_ape,
        )

        self.tabular_encoder = TabularEncoder(
            in_dim=tabular_num_features,
            out_dim=embed_dim,
            num_layers=tabular_num_hidden_layers,
            layer_dims=tabular_layer_dims,
            act_layer=nn.ReLU,
            dropout_prob=tabular_dropout_prob,
            norm_layer=nn.BatchNorm1d,
            mask_proportion=mask_proportions["tabular"],
            mask_token=mask_token,
        )

        self.decoder = Decoder(
            num_heads=decoder_num_heads,
            num_blocks=decoder_num_blocks,
            mask_proportion=mask_proportions,
            mask_token=mask_token,
        )

        if mask_proportions:
            self.img_expand_features = nn.Conv2d(
                in_channels=img_chans,
                out_channels=embed_dim,
                kernel_size=swin_patch_size,
                stride=swin_patch_size,
            )
            self.text_expand_features = nn.Linear(in_features=1, out_features=embed_dim)
            self.tab_expand_features = nn.Linear(
                in_features=tabular_num_features, out_features=embed_dim
            )

    def forward(self, img_tensor, text_tensor, text_mask, tab_tensor):
        embed_img = self.swin_encoder(img_tensor)
        embed_text = self.roberta_encoder(text_tensor, text_mask)
        embed_tab = self.tabular_encoder(tab_tensor)

        # if mask proportions specified, forward() returns the masked input,
        # the mask (except the text one), and the embeddings
        if self.mask_proportions:
            masked_img, img_mask, embed_img = embed_img
            masked_text, embed_text = embed_text
            masked_tab, tab_mask, embed_tab = embed_tab

        # concat along sequence dimension
        all_embed = torch.cat((embed_tab, embed_text, embed_img), dim=1)

        if self.mask_proportions:
            # expand each of the feature dims to the encoder's embed dim
            masked_img = self.img_expand_features(masked_img).flatten(2).transpose(1, 2)
            masked_text = self.text_expand_features(masked_text)
            masked_tab = self.tab_expand_features(masked_tab)

            all_masked = torch.cat((masked_tab, masked_text, embed_img), dim=1)

            return
            y = self.decoder(
                dec_input=all_masked,
                enc_output=all_embed,
            )

            return y, img_mask, text_mask, tab_mask

        y = self.decoder(img_tensor, enc_output=all_embed)

        return y

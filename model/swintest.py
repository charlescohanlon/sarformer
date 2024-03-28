# from SwinTransformer.models import swin_transformer_v2 as st2
import model.swintransformerv2 as st2
import torch.nn as nn
import torch

img_size = 224 # 224 - 7 window size, 256 - 8 window size
patch_size = 4
patches_resolution = (img_size // patch_size, img_size // patch_size)
 
embed_dim = 96
depths = [2, 2, 6, 2]  # Depths of each Swin Transformer layer
num_heads = [3, 6, 12, 24] 

layers = []
for i_layer in range(len(depths)):
    layer = st2.BasicLayer(
        dim=int(embed_dim * 2 ** i_layer),
        input_resolution=(patches_resolution[0] // (2 ** i_layer), patches_resolution[1] // (2 ** i_layer)),
        depth=depths[i_layer],
        num_heads=num_heads[i_layer],
        window_size=7,
        mlp_ratio=4.,  # Default
        qkv_bias=True,  # Default 
        norm_layer=nn.LayerNorm,  # Default 
        downsample=st2.PatchMerging if (i_layer < len(depths) - 1) else None, 
        use_checkpoint=False,  # Default
        pretrained_window_size=0)  # Default
    layers.append(layer)

# Propagate patch embeddings through the Swin Transformer layers
x = torch.randn(1, patches_resolution[0] * patches_resolution[1], embed_dim)  
for layer in layers:
    x = layer(x)


output_feature_dim = x.shape[-1]
print(output_feature_dim)
print(x.shape)
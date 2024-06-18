import torch
from PIL import Image
import numpy as np
import rasterio
from sarformer import SARFormer
import torch.nn.functional as F

model = SARFormer(512, 768, mask_proportions={"swin":0.5,"bert":0.5,"tabular":0.5})

def load_and_preprocess_images(dem_path, naip_path):
    with rasterio.open(dem_path) as dem_file:
        dem = dem_file.read(1)  # Read first band

    with rasterio.open(naip_path) as naip_file:
        naip_img = naip_file.read([1, 2, 3])  # Read RGB bands
        naip_img = np.transpose(naip_img, (1, 2, 0))  # Change to HWC format

    # Normalize and convert to tensor
    dem = torch.tensor(dem, dtype=torch.float32)
    naip_img = torch.tensor(naip_img, dtype=torch.float32)
    dem = (dem - dem.mean()) / dem.std()
    naip_img = naip_img / 255.0

    # Stack DEM and RGB to form 4-channel input
    combined_img = torch.cat([naip_img, dem.unsqueeze(-1)], dim=-1)
    combined_img = combined_img.permute(2, 0, 1).unsqueeze(0)  # Change to BCHW format for the model
    
    # 1 x 4 x 512 x 512
    return combined_img

#image_tensor = load_and_preprocess_images(r'C:\Users\camer\OneDrive\Documents\AI4SAR\DL4SAR\loaders\TestInputs\DEM_643554.tif', \
#                                          r'C:\Users\camer\OneDrive\Documents\AI4SAR\DL4SAR\loaders\TestInputs\NAIP_643554.tif')

# text = torch.randn(1, 100) # Batch size of 1, 100 features
text = [
    "A scent of sagebrush and earth: I find a faint scent of sagebrush and earth in the air...",
]

tabular_tensor = torch.randn(1, 10)  # Batch size of 1, 10 features


#swin_embed = model.swin_encoder(image_tensor)
bert_embed = model.bert_encoder(text)
tab_embed = model.tabular_encoder(tabular_tensor)

if model.mask_proportions:
    #swin_masked, swin_embed = swin_embed
    bert_masked, bert_embed = bert_embed
    tab_masked, tab_embed = tab_embed

# Fake for now
swin_masked = torch.randn(1, 2048, 768)
swin_embed = torch.randn(1, 64, 768)


cat_embed = torch.cat((tab_embed, swin_embed, bert_embed), dim=1).permute(0, 2, 1)
if model.mask_proportions:
    # NOTE: swin_masked is (batch, (512 / 4)^2, 96). We'll need to pad the embedding 
    # dims of tab_masked and bert_masked to concat them
    bert_masked = F.pad(bert_masked, (0, 768 - bert_masked.size(2)))
    tab_masked = F.pad(tab_masked, (0, 768 - tab_masked.size(2)))

    cat_masked = torch.cat((tab_masked, swin_masked, bert_masked), dim=1).permute(0, 2, 1)

    # "embed" is the shrunken embeddings (input to the cross-attention sublayer)
    # "masked" is the input with the mask applied (input to the mhsa sublayer)
    print(cat_embed.shape)
    print(cat_masked.shape)
    
    output = model.decoder(inputs = {"embed": cat_embed, "masked": cat_masked})

else:
    output = model.decoder(input_dim = 0, inputs = {"embed": cat_embed})

print(output)

print("output size:")
print(output.size())

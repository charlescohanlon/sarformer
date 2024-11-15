# DOCUMENTATION

## Random Notes

Pre-train idea:
Use a pre-trained text encoder (e.g., T5) to create latent embeddings for sequence input
1. Cut out NAIP/DEM tifs at 1785 (893 + 893 - 1)
2. Crop random 893 (447 + 447 - 1) 
    - supposed to produce a simulated observation that takes place in same viscinity
    - select crop based on some kind of normal distribution complement? (that makes edges more likely)
3. Simulate prob agent from random start within 893 crop
4. Crop 447 w/ end point at center
During pre-training
    - freeze text encoder
    - select random SAR case to select cut out image
        - take corresponding data and add noise then use as conditioning
Fine-tune on sar
    - still weight real cases in the loss


### Todo:
- Cameron:
- Matthew:
    - research noising as an augmentation
    - write augmenter code
- Charles: 
    - write shapefile script
    - write mae conv code
    - test t5 encoder

- Compute NAIP mean and standard deviation on 447 images
- Random crop with normal distribution
- Download new weather data values
    - debug segfault issue
- Bucket sar, time, and weather values
    - how are we bucketing (someone should review code)
        - how many and what quantiles?
    - bucket numerical sar columns (find Cameron source for this?)
- Implement CaptionTransform and StructuredDataTransform
    - CaptionTransform should create the sentence in load() and shuffle the ordering
        - choose sentence fill-in-the-blanks at random per column for a given dataset
        - generate 10 per column (someone should review code)
    - StructuredDataTransform should include a random attention mask
        - also include IPP position in pixel distance
            - check how many IPPs fall outside image to get idea
            - only if within image? outside image too?
- Select SARFormer val split
    - select from cases where nothing was imputed
- Train SARFormer
    - weight real observations in the loss (find source for this?)

### Experiments and Visualizations

- attention visualization
    - take attention weights from encoder's first attention block?
    - visualize attention of IPP wrt patches of the image
        - We would expect intutively for IPP to be a strong signal
- dimensionality reduction
    - take latent space from somewhere?
- domain adaptation: leave an SAR set out and see how well it does w/ zero-shot

#### To Ablate:

- modalities
- pixel distance weighting of loss
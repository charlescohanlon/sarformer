# DOCUMENTATION

## Random Notes

Mistakes from last vqvae run: wrong mean and std, too few warmup epochs (use 100), many images were all or mostly black, validation computed too often, 

Idea: perceptual loss w/ some kind of sparse image model

### Todo:

- Fit probabilistic agent model
    - Select cases that have IPP and found point within the image, also that have subject categories
    - run on 893x893 images, but constrain to 447x447 center crop 
- Run inference w/ probabilistic agent model
    - run once on all unlabeled cases and produce 447x447 crops with the resulting label at the center
- Train VQVAE on new images
    - need to compute NAIP mean and standard deviation on 447 images
    - figure out why codebook usage frequency works
- Pre-tokenize images
    - Generate aug params for 1000 crops each
    - Move tokens to OSN (object store)
    - write Tok load() to get from OSN
        - dataloader should choose UID then augmentation index at random
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
        - noise as augmentation?
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
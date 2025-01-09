# DOCUMENTATION

## Experiements
parameterize a gaussian with output of the network
 - use the distribution of distances from found

see where we can get with tabular only

rgb + depth only

regression on the point only



### Todo:
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
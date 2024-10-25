# DOCUMENTATION

## Random Notes

TODO:
- Look for missing columns
    - include columns for whether a value was imputed or not (See that youtube vid)
- Download new images
    - make sure all rows (excluding Novia Scotia) is a subset
- Train VQVAE on new images
    - need to compute NAIP mean and standard deviation on 447 images
    - Val split doesn't need to be same as for SARFormer
- Pre-tokenize images
    - Generate aug params for 100 crops each (excluding the val split)
    - Include new output 1 positions
- Download new weather data values
    - Need to determine how they're spatially and temporally aggregated
    - Ask Cameron to bucket them
- Implement CaptionTransform and StructuredDataTransform (Do this weekend)
    - CaptionTransform should open create the sentence in load() and shuffle the ordering
        - use synonyms
    - StructuredDataTransform should include a random attention mask
        - also include IPP position in pixel distance (even if it's outside the random crop)
- Select SARFormer val split
- Train SARFormer


synthetic data: (extending the distribution)
1. Cluster the distributions based on some kind of location
    - e.g., sar set (I'm guessing most sets happen around the same area)
2. Sample the categorical distributions for each column N times
3. Train a classifier to map a sampled row to a behavior profile
4. Execute behavior profile w/ agent on same image as ...?
    - run agent for timesteps equivalent to sampled duration
    - how to condition on weather?
    - want to get new images?
5. Use as synthetic missing person case


### Reducing Noise in Dataset

[Data curation](https://atlan.com/data-curation-in-machine-learning/)

The data we evaluate on has to be ALL real so no generated times or dates either

### Experiments and Visualizations

- attention visualization
    - take attention weights from encoder's first attention block?
    - visualize attention of IPP wrt patches of the image
        - We would expect intutively for IPP to be a strong signal
- dimensionality reduction
    - take latent space from somewhere?
- domain adaptation: leave an SAR set out and see how well it does w/ zero-shot
- weight the real data in the loss?

#### To Ablate:

- modalities
- pixel distance weighting of loss
- model size
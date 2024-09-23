# DOCUMENTATION

## Random Notes

### Projects

Cameron: train mlp tokenizer for structured_data
- I'll need to request more Inodes
- Should validate the weather columns we use so we can cite them
- include columns for whether a value was imputed or not (See that youtube vid)

Charles: VQ-VAE for RGB and depth instead of DiVAE b/c we don't really care about reconstruction quality, we're more worried about the quality of the representation and VQ-VAE is less expensive to train
- Need to regenerate the SAR rgb and depth images first
- Get NAIP and DEM equivalent for nonUS

Use Norm Attention in Unet?

Depth data has inf

Should log hparams and give runs unique IDs or something

The data we evaluate on has to be ALL real so no generated times or dates either

Explanation:
So the other day I was thinking and as you guys know pre-training with a masked modeling objective is only a bandaid,
ideally we have all 1 million cases fully labeled. With 4M they pseudo label using powerful specialized models, then
take all of that as input and predict it all to produce generative behavior. If we were to follow suit we'd be wasting
a lot of time teaching our model to be a generative one. At first, this made sense because in an abstract way the model
is also learning correlations between various features in the input. But what if, we don't use a masked modeling objective
at all. Instead we create coordinate labels for all the synthetic and unlabeled cases too. Then we just use the heatmap
generation objective on all of it. For the synthetic and unlabeled cases the model is still essentially learning an 
association just with possible behaviors (of a lost person) and not with other features of the input. The question then
becomes, how do we pseudo label the cases without knowing where they took place (or having a model that can predict that)?
I thought of two ways,
1. we use the agent-based approach that one paper was talking about
    - for the unlabeled SAR cases we need some way of getting a center coordinate for the case
2. we bucket the labeled data observations somehow, fit a bivariate distribution on them (with something like MLE) then
sample from that for every synthetic case
- we're also going to weight the real data in the loss

### Reducing Noise in Dataset

Review how we did each of the imputations

[Data curation](https://atlan.com/data-curation-in-machine-learning/)

### Stuff to Try

- analytic approach to choosing eval split
- weight labeled data data in loss function
- regenerate MC @ 224
    - stratified sampling of different regions
    - regenerate corresponding LLM prompts

### Experiments and Visualizations

- attention visualization
    - take attention weights from encoder's first attention block?
- dimensionality reduction
    - take latent space from somewhere?
- domain adaptation: leave an SAR set out and see how well it does w/ zero-shot

#### To Ablate:

- modalities
- pixel distance weighting of loss
- model size



## Tokenization

### Metadata (Strutured Data) [NEEDS TO BE UPDATED]

Building off of 4M's approach, which uses several vX=Y (0<=X<=4, 0<=Y<=999) tokens to communicate the value of metadata, we're using v0=(something) and v1=(hexadecimal of float32 value). Where v0 denotes the type of metadata, and v1 indicates the value of said metadata attribute. v0 ranges from [0, # types of metadata].


Example of three metadata representations for 3 types of metadata:
```
v0=2 v1=A8210D4A v0=0 v1=AFFEB3A2 v0=1 v1=E34B0C9F
```

Order of a pair of v0 and v1 doesn't matter (and is randomized), but a v0 is always followed by a v1 that denotes it's value.

We're using this approach because it allows us to represent arbitrarily large floating point values without a large vocabulary size and without increasing the sequence length dramatically. It seems 4M used values in the range [0, 999], which gives their metadata attributes 1000 possible values. They included a different token in the vocabulary for each possible value (see fourm/utils/tokenizer/trained/text_tokenizer_4m_wordpiece_30k.json). This route is unsatisfactory if we want the ability to represent arbitrary floating point values (we'd need a vocab size of over 10^38). Another potential route is to only have tokens for individual digits (base 10) but then sequence length would increase with number precision (and magnitude) very quickly.

Our solution is to use hex to represent the IEEE 754 standard for the floating point numbers. 32-bit floating point fixes each value to a sequence length of 4 bytes (8 hex digits). We then provide special tokens for every combination of 2 hex digits (i.e., one byte). So a 32-bit floating point value is represented by only 4 tokens, and the vocab size only increased by 256.

It's worth noting that varying the number system base used to represent values will shift the sequence length and token counts. E.g., base 2 => longer sequence and fewer tokens, base 36 => shorter sequence and more tokens.

This idea is kind of similar to byte-level BPE.
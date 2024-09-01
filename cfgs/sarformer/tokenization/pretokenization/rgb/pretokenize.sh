HOME=/u/$USER
source $HOME/.bashrc # to add micromamba to path
cd $HOME/sarformer
micromamba activate -n sarformer

python save_vq_tokens.py \
    --tokenizer_id "checkpoint-39" \
    --tokenizers_root "output/tokenization/rgb/ViTB-UNetP4_16k_224_predx0" \
    --data_root "/scratch/bdej/cohanlon/untokenized" \
    --split "train" \
    --input_size 224 \
    --task "rgb" \
    --mask_value 0 \
    --resample_mode "cubic" \
    --verbose True \
    --device "cuda" \
    --folder_suffix "_rgb_toks" \
    --batch_size 96 \
    --dry_run False # creates but doesn't save tokens
HOME=/u/$USER
source $HOME/.bashrc # to add micromamba to path
cd $HOME/sarformer
micromamba activate -n sarformer

python train_wordpiece_tokenizer.py \
    --text_files /scratch/bdej/cohanlon/all_text_including_nonUS.txt \
    --save_file $HOME/sarformer/fourm/utils/tokenizer/trained/tokenizer_inc_nonUS_lower.json \
    --vocab_size 30000 \
    --num_sentinels 200 \
    --num_metadata 21 \
    --lowercase
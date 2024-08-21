import json
import os
from collections import defaultdict
from typing import Optional, Union, List
from itertools import product

from tokenizers import AddedToken, decoders, trainers
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer


def generate_sentinel_tokens(num=100, start_id=0):
    tokens = [
        AddedToken(content=f"[S_{i}]", single_word=True, normalized=False)
        for i in range(start_id, num + start_id)
    ]

    return tokens


def generate_coord_tokens(bins=1000):
    """Extra tokens that are used for bounding box coordinates,
    xmin, ymin, xmax, ymax, but also other modalities like color
    maps, metadata, or poses.
    """
    tokens = []
    coords_str = ["v0={}", "v1={}", "v2={}", "v3={}"]

    for s in coords_str:
        for i in range(bins):
            tokens.append(
                AddedToken(content=s.format(i), single_word=True, normalized=False)
            )

    return tokens


def generate_metadata_tokens(num=0, hex_tok_len=2):
    tokens = []
    type_token = "v0={}"
    for i in range(num):
        tokens.append(
            AddedToken(content=type_token.format(i), single_word=True, normalized=False)
        )

    tokens.append(AddedToken(content="v1=", single_word=False, normalized=False))
    hex_digits = "0123456789ABDCDEF"
    # hex_tok_len = 2 => [01], [02], ..., [0F], [10], [11], ..., [FF]
    for c in product(hex_digits, repeat=hex_tok_len):
        tokens.append(
            AddedToken(content=f"[{''.join(c)}]", single_word=False, normalized=False)
        )

    # year tokens
    for i in range(1970, 2025):  # NOTE: these years are kind of arbitrary
        tokens.append(
            AddedToken(content=f"[{i}]-", single_word=False, normalized=False)
        )

    # month tokens
    for i in range(1, 13):
        tokens.append(
            AddedToken(content=f"-[{i:02}]-", single_word=False, normalized=False)
        )

    # day tokens
    for i in range(1, 32):
        tokens.append(
            AddedToken(content=f"-[{i:02}]", single_word=False, normalized=False)
        )

    # hour tokens
    for i in range(24):
        tokens.append(
            AddedToken(content=f"[{i:02}]:", single_word=False, normalized=False)
        )

    # minute tokens
    for i in range(60):
        tokens.append(
            AddedToken(content=f":[{i:02}]:", single_word=False, normalized=False)
        )

    # second tokens
    for i in range(60):
        tokens.append(
            AddedToken(content=f":[{i:02}]", single_word=False, normalized=False)
        )

    return tokens


def generate_object_class_tokens(dataset="coco"):
    with open(os.path.join(os.path.dirname(__file__), "object_classes.json")) as f:
        object_classes = json.load(f)[dataset]

    tokens = [
        AddedToken(content=class_name, single_word=True, normalized=True)
        for class_name in object_classes
    ]

    return tokens


def train_unified_wordpiece_tokenizer(
    files,
    vocab_size,
    sentinel_tokens: List[Union[str, AddedToken]] = None,
    coord_tokens: List[Union[str, AddedToken]] = None,
    object_class_tokens: List[Union[str, AddedToken]] = None,
    metadata_tokens: List[Union[str, AddedToken]] = None,
    unk_token: Union[str, AddedToken] = "[UNK]",
    pad_token: Union[str, AddedToken] = "[PAD]",
    sos_token: Union[str, AddedToken] = "[SOS]",
    eos_token: Union[str, AddedToken] = "[EOS]",
    additional_special_tokens: List[Union[str, AddedToken]] = None,
    min_frequency=0,
    clean_text: bool = True,
    handle_chinese_chars: bool = True,
    strip_accents: Optional[bool] = None,
    lowercase: bool = True,
    wordpieces_prefix: str = "##",
    show_progress=True,
):
    tokenizer = Tokenizer(WordPiece(unk_token=str(unk_token)))

    tokenizer.normalizer = BertNormalizer(
        clean_text=clean_text,
        handle_chinese_chars=handle_chinese_chars,
        strip_accents=strip_accents,
        lowercase=lowercase,
    )
    tokenizer.pre_tokenizer = BertPreTokenizer()
    tokenizer.decoder = decoders.WordPiece(prefix=wordpieces_prefix)

    special_tokens = []
    special_tokens.append(pad_token)
    special_tokens.append(unk_token)
    special_tokens.append(sos_token)
    special_tokens.append(eos_token)

    if sentinel_tokens is not None:
        special_tokens.extend(sentinel_tokens)
    if coord_tokens is not None:
        special_tokens.extend(coord_tokens)
    if object_class_tokens is not None:
        special_tokens.extend(object_class_tokens)
    if metadata_tokens is not None:
        special_tokens.extend(metadata_tokens)
    if additional_special_tokens is not None:
        special_tokens.extend(additional_special_tokens)

    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=show_progress,
        continuing_subword_prefix=wordpieces_prefix,
        special_tokens=special_tokens,
    )

    if isinstance(files, str):
        files = [files]

    tokenizer.train(files, trainer=trainer)

    return tokenizer


def get_sentinel_to_id_mapping(tokenizer, match_str="[S_"):
    sentinel_tokens = {
        k: v for k, v in tokenizer.get_vocab().items() if k.startswith(match_str)
    }
    # Extract the sentinel token id, the id is of the form "[S_0]", "[S_1]", etc.
    sentinel_to_id = {
        int(k.split("_")[1][:-1]): v
        for k, v in sorted(sentinel_tokens.items(), key=lambda x: x[1])
    }
    return sentinel_to_id


def split_by_sentinel(seq_ids, sentinel_ids):
    splits = defaultdict(list)
    cur_sentinel = None
    for token in seq_ids:
        if token in sentinel_ids:
            cur_sentinel = token
        else:
            splits[cur_sentinel].append(token)

    return splits


def merge_span_masking(input_seq, decoder_seq, sentinel_ids):
    decoder_splits = split_by_sentinel(decoder_seq, sentinel_ids)
    out_seq = []
    for token in input_seq:
        if token in sentinel_ids:
            out_seq.extend(decoder_splits[token])
        else:
            out_seq.append(token)
    return out_seq

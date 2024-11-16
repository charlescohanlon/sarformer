# Copyright 2024 EPFL and Apple Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse
from src.utils.tokenizer import train_unified_wordpiece_tokenizer
from src.utils.tokenizer import (
    generate_sentinel_tokens,
    generate_coord_tokens,
    generate_object_class_tokens,
    generate_metadata_tokens,
)


def get_args():
    parser = argparse.ArgumentParser(
        "Train unified WordPiece tokenizer", add_help=False
    )
    parser.add_argument(
        "--text_files",
        type=str,
        default="/datasets/imagenet_multitask/metadata/all_captions_BLIP.txt",
        help="Files to train the tokenizer on, separated by a double dash '--'",
    )
    parser.add_argument(
        "--save_file",
        type=str,
        default="fourm/utils/tokenizer/trained/default_tokenizer.json",
        help="Path to the saved tokenizer. Can then be loaded using Tokenizer.from_file(path).",
    )
    parser.add_argument(
        "--vocab_size", type=int, default=30_000, help="Vocabulary size"
    )
    parser.add_argument(
        "--num_sentinels", type=int, default=200, help="Number of sentinel tokens"
    )
    parser.add_argument(
        "--coord_bins",
        type=int,
        default=0,  # Not using these
        help="Number of coordinate bins (for detection)",
    )
    parser.add_argument(
        "--num_metadata",
        type=int,
        default=0,
        help="Number of types of metadata to be tokenized",
    )
    parser.add_argument(
        "--hex_tok_len",
        type=int,
        default=2,
        help="Length of hex tokens for metadata",
    )
    parser.add_argument(
        "--object_classes",
        type=str,
        default="none",  # Not using these
        choices=["none", "coco"],
        help="Special tokens for detection instances (e.g., instance class names from the COCO dataset)",
    )
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--no_lowercase", action="store_false", dest="lowercase")
    parser.set_defaults(lowercase=True)
    return parser.parse_args()


# TODO: check AddedToken args and whether training is MP or not, also review WordPiece
def train_tokenizer(args):

    files = args.text_files.split("--")
    # Get special tokens
    sentinel_tokens = generate_sentinel_tokens(num=args.num_sentinels)
    coord_tokens = generate_coord_tokens(bins=args.coord_bins)
    metadata_tokens = generate_metadata_tokens(
        num=args.num_metadata, hex_tok_len=args.hex_tok_len
    )
    if args.object_classes == "none":
        object_class_tokens = None
    else:
        object_class_tokens = generate_object_class_tokens(args.object_classes)

    print(f"Training tokenizer on files: {files}")

    # Train tokenizer
    tokenizer = train_unified_wordpiece_tokenizer(
        files=files,
        vocab_size=args.vocab_size,
        sentinel_tokens=sentinel_tokens,
        coord_tokens=None,  # Not using these
        object_class_tokens=None,  # Not using these
        metadata_tokens=metadata_tokens,
        lowercase=args.lowercase,
    )

    # Create directory of target file if it doesn't exist
    os.makedirs(os.path.dirname(args.save_file), exist_ok=True)
    tokenizer.save(path=args.save_file)

    print(f"Tokenizer saved to: {args.save_file}!")


if __name__ == "__main__":
    args = get_args()
    train_tokenizer(args)

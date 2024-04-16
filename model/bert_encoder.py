from collections import OrderedDict
from typing import Sequence
from transformers import AutoTokenizer, BertConfig, BertModel as TransformersBertModel
import torch
from torch import nn


class BertEncoder(nn.Module):
    """
    add_pooling_layer (bool): whether to add a pooling layer.
    num_layers_of_embedded (int): number of layers of the embedded model (defaults to 12)
    """

    def __init__(
        self, add_pooling_layer: bool = False, num_layers_of_embedded: int = 12
    ):
        super().__init__()
        config = BertConfig.from_pretrained("bert-base-cased")

        self.model = TransformersBertModel.from_pretrained("bert-base-cased")
        self.hidden_size = config.hidden_size  # should be 768
        self.num_layers_of_embedded = num_layers_of_embedded

    def forward(self, x) -> dict:
        mask = x["attention_mask"]

        # Apply masking logic
        if self.training:  # Apply masking only during training
            x["input_ids"], mask = self.mask_input(x["input_ids"], mask)

        outputs = self.model(
            input_ids=x["input_ids"],
            attention_mask=mask,
            output_hidden_states=True,
        )

        # outputs has 13 layers, 1 input layer and 12 hidden layers
        encoded_layers = outputs.hidden_states[1:]

        features = torch.stack(encoded_layers[-self.num_layers_of_embedded :], 1).mean(
            1
        )

        features = features / self.num_layers_of_embedded

        if mask.dim() == 2:
            embedded = features * mask.unsqueeze(-1).float()
        else:
            embedded = features

        results = {"embedded": embedded, "masks": mask, "hidden": encoded_layers[-1]}
        return results

    def mask_input(self, input_ids, mask):
        with torch.no_grad():
            num_tokens_keep = int(mask.sum() * (1.0 - self.mask_proportion))
            indices_to_mask = torch.randperm(input_ids.numel())[:num_tokens_keep]
            mask_tensor = torch.zeros_like(input_ids, dtype=torch.bool)
            mask_tensor.view(-1)[indices_to_mask] = True
            input_ids[mask_tensor] = self.mask_token_id
            mask[mask_tensor] = 0  # Set corresponding attention_mask to 0 for masked tokens
        return input_ids, mask


class BertModel(nn.Module):
    """
    max_tokens (int): maximum number of tokens to be used for BERT (defaults to 256)
    add_pooling_layer (bool): whether to adding pooling layer
                              in bert encoder (defaults to False)
    num_layers_of_embedded (int): number of layers of the embedded model
                                  (defaults to 12)
    """

    def __init__(
        self,
        max_tokens: int = 256,
        add_pooling_layer: bool = False,
        num_layers_of_embedded: int = 12,
        mask_proportion: float = 0.15,
        mask_token_id: int = 103,
    ):

        super().__init__()
        self.max_tokens = max_tokens
        self.mask_proportion = mask_proportion
        self.mask_token_id = mask_token_id

        print(add_pooling_layer)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.language_backbone = nn.Sequential(
            OrderedDict(
                [
                    (
                        "body",
                        BertEncoder(
                            "bert-base-cased",
                            num_layers_of_embedded=num_layers_of_embedded,
                        ),
                    )
                ]
            )
        )

    def forward(self, text: Sequence[str]) -> dict:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tokenized = self.tokenizer(
            text,
            max_length=self.max_tokens,
            padding="longest",
            return_tensors="pt",
            truncation=True,
        ).to(device)

        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask

        tokenizer_input = {"input_ids": input_ids, "attention_mask": attention_mask}

        language_dict_features = self.language_backbone(tokenizer_input)
        return language_dict_features


bert_model = BertModel()

text = ["This is a sample sentence.", "Another example sentence."]

output = bert_model(text)

print(output["embedded"])

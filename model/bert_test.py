from collections import OrderedDict
from typing import Sequence
from transformers import AutoTokenizer, BertConfig, BertModel as TransformersBertModel
import torch
import torch.optim as optim
from torch import nn
from typing import Optional
import mlflow


class BertEncoder(nn.Module):
    """
    num_layers_of_embedded (int): number of layers of the embedded model (defaults to 12)
    """

    def __init__(
        self, num_layers_of_embedded: int = 12
    ):
        super().__init__()
        config = BertConfig.from_pretrained("bert-base-cased")

        self.model = TransformersBertModel.from_pretrained("bert-base-cased")
        self.hidden_size = config.hidden_size  # should be 768
        self.num_layers_of_embedded = num_layers_of_embedded

    def forward(self, x) -> dict:
        outputs = self.model(
            input_ids=x["input_ids"],
            output_hidden_states=True,
        )

        # outputs has 13 layers, 1 input layer and 12 hidden layers
        encoded_layers = outputs.hidden_states[1:]

        features = torch.stack(encoded_layers[-self.num_layers_of_embedded :], 1).mean(
            1
        )
        features = features / self.num_layers_of_embedded
        return features


class BertModel(nn.Module):
    """
    max_tokens (int): maximum number of tokens to be used for BERT (defaults to 256)
    num_layers_of_embedded (int): number of layers of the embedded model
                                  (defaults to 12)
    masked_proportion (float): proportion of input to be masked
    """

    def __init__(
        self,
        max_tokens: int = 256,
        num_layers_of_embedded: int = 12,
        masked_proportion: Optional[float] = 0
    ):

        super().__init__()
        self.max_tokens = max_tokens
        self.masked_proportion = masked_proportion
        self.num_layers_of_embedded = num_layers_of_embedded

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(device)

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        print(self.tokenizer)
        self.language_backbone = nn.Sequential(
            OrderedDict(
                [
                    (
                        "body",
                        BertEncoder(
                            num_layers_of_embedded=num_layers_of_embedded,
                        ),
                    )
                ]
            )
        )

    def forward(self, text: Sequence[str]):
        # Tokenize input text
        tokenized = self.tokenizer(
            text,
            max_length=self.max_tokens,
            padding="longest",
            return_special_tokens_mask=True,
            return_tensors="pt",
            truncation=True,
        )

        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask
        x = input_ids

        if self.masked_proportion is not None:
            num_tokens = attention_mask.shape[1] # needs to be attention mask to avoid masking padded tokens
            num_masked_tokens = int(self.masked_proportion * num_tokens)
            masked_indices = torch.randperm(num_tokens)[:num_masked_tokens]
            input_ids[:, masked_indices] = self.tokenizer.mask_token_id
            attention_mask[:, masked_indices] = 0

            mask_indices_kept = (input_ids != self.tokenizer.mask_token_id)
        
            # Extract non-masked values from input_ids
            x = input_ids[mask_indices_kept]
            x = x.view(input_ids.shape[0], -1)

        tokenizer_input = {"input_ids": x, "attention_mask": attention_mask}

        bert_embed = self.language_backbone(tokenizer_input)

        if self.masking_proportion is not None:
            masked_input = input_ids.clone()

            masked_input[:, masked_indices] = self.tokenizer.mask_token_id

            return masked_input, bert_embed

        return bert_embed
    
from collections import OrderedDict
from typing import Sequence
from transformers import AutoTokenizer, BertConfig, BertModel as TransformersBertModel
import torch
import torch.optim as optim
from torch import nn
from typing import Optional
import mlflow

class Backbone(nn.Module):
    def __init__(self, embedding_layers: int = 12):
        super().__init__()
        config = BertConfig.from_pretrained("bert-base-cased")

        self.model = TransformersBertModel.from_pretrained("bert-base-cased")
        self.hidden_size = config.hidden_size  # should be 768
        self.embedding_layers = embedding_layers

    def forward(self, x) -> torch.Tensor:
        outputs = self.model(
            input_ids=x["input_ids"],
            attention_mask=x["attention_mask"],
            output_hidden_states=True,
        )

        # outputs has 13 layers, 1 input layer and 12 hidden layers
        encoded_layers = outputs.hidden_states[1:]

        features = torch.stack(encoded_layers[-self.embedding_layers:], 1).mean(1)
        features = features / self.embedding_layers
        return features
    
class BertEncoder(nn.Module):
    def __init__(
        self,
        max_tokens: int = 1024,
        embedding_layers: int = 12,
        mask_proportion: Optional[float] = 0
    ):

        super().__init__()
        self.max_tokens = max_tokens
        self.masked_proportion = mask_proportion
        self.embedding_layers = embedding_layers

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(device)
        self.device = device 

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.language_backbone = nn.Sequential(
            OrderedDict(
                [
                    (
                        "body",
                        Backbone(
                            embedding_layers=embedding_layers,
                        ),
                    )
                ]
            )
        ).to(device)

    def forward(self, text: Sequence[str]):
        tokenized = self.tokenizer(
            text,
            max_length=512,#self.max_tokens,
            padding='max_length',
            return_special_tokens_mask=True,
            return_tensors="pt",
            truncation=True,
        )

        input_ids = tokenized.input_ids.to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device)

        #print(input_ids.shape)
        #print(attention_mask.shape)
        if self.masked_proportion is not None:
            num_tokens = attention_mask.shape[1]
            num_masked_tokens = int(self.masked_proportion * num_tokens)
            masked_indices = torch.randperm(num_tokens)[:num_masked_tokens]
            input_ids[:, masked_indices] = self.tokenizer.mask_token_id
            attention_mask[:, masked_indices] = 0

        #print(input_ids.shape)
        #print(attention_mask.shape)
        tokenizer_input = {"input_ids": input_ids, "attention_mask": attention_mask}
        bert_embed = self.language_backbone(tokenizer_input)

        if self.masked_proportion is not None:
            return bert_embed, bert_embed

        return bert_embed

"""
bert_model = BertEncoder()
text = ["This is a sample sentence.", "Another example sentence."]
output = bert_model(text)
print(output)

bert_model = BertEncoder(mask_proportion=0.5)
text = ["This is a sample sentence.", "Another example sentence."]
output = bert_model(text)
print(output)
"""
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

    def __init__(self, add_pooling_layer: bool = False, num_layers_of_embedded: int = 12):
        super().__init__()
        config = BertConfig.from_pretrained('bert-base-cased')
        
        self.model = TransformersBertModel.from_pretrained('bert-base-cased')
        self.hidden_size = config.hidden_size # should be 768
        self.num_layers_of_embedded = num_layers_of_embedded

    def forward(self, x) -> dict:
        mask = x['attention_mask']

        outputs = self.model(
            input_ids=x['input_ids'],
            attention_mask=mask,
            output_hidden_states=True,
        )

        # outputs has 13 layers, 1 input layer and 12 hidden layers
        encoded_layers = outputs.hidden_states[1:]

        features = torch.stack(encoded_layers[-self.num_layers_of_embedded:], 1).mean(1)
        
        features = features / self.num_layers_of_embedded
        
        if mask.dim() == 2:
            embedded = features * mask.unsqueeze(-1).float()
        else:
            embedded = features

        results = {
            'embedded': embedded,
            'masks': mask,
            'hidden': encoded_layers[-1]
        }
        return results


class BertModel(nn.Module):
    """
        max_tokens (int): maximum number of tokens to be used for BERT (defaults to 256)
        add_pooling_layer (bool): whether to adding pooling layer 
                                  in bert encoder (defaults to False)
        num_layers_of_embedded (int): number of layers of the embedded model
                                      (defaults to 12)
    """

    def __init__(self, max_tokens: int = 256, add_pooling_layer: bool = False, num_layers_of_embedded: int = 12):

        super().__init__()
        self.max_tokens = max_tokens

        print(add_pooling_layer)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        self.language_backbone = nn.Sequential(OrderedDict([('body', BertEncoder('bert-base-cased',
                                 num_layers_of_embedded=num_layers_of_embedded))]))


    def forward(self, text: Sequence[str]) -> dict:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # https://huggingface.co/docs/transformers/v4.39.3/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.batch_encode_plus
        tokenized = self.tokenizer.batch_encode_plus(
            text,
            max_length=self.max_tokens,
            padding='longest',
            return_special_tokens_mask=True,
            return_tensors='pt',
            truncation=True).to(device)
        
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask

        tokenizer_input = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

        language_dict_features = self.language_backbone(tokenizer_input)
        return language_dict_features

# Initialize the model
bert_model = BertModel()

# Define input text
text = ["This is a sample sentence.", "Another example sentence."]

# Perform forward pass
output = bert_model(text)

# Print the embedded features
print(output['embedded'])
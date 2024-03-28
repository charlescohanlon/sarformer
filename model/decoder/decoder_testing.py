import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm # progress bar
# import data loader and other relevant files

class Decoder4M(nn.Module):
    def __init__(self, hidden_dim, num_decoder_layers, num_attention_heads, window_size):
        super(Decoder4M, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_decoder_layers = num_decoder_layers
        self.num_attention_heads = num_attention_heads
        self.window_size = window_size
        
        # Shared positional embeddings (not sure how/if this will be implemented)
        self.positional_embedding = nn.Parameter(torch.randn(window_size * window_size, hidden_dim))
        
        # Shared modality embeddings (also not sure about this with only 3 modalities)
        self.modality_embedding = nn.Parameter(torch.randn(3, hidden_dim))
        
        # Transformer decoder (based on "Attention is All You Need" paper)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_attention_heads),
            num_layers=num_decoder_layers
        )
        
        # Shared output embedding (final linear layer)
        self.output_embedding = nn.Linear(hidden_dim, 1)
        
    def forward(self, concatenated_input, sequence_mask=None):
        # Add modality embeddings (just adding for now, will likely change later)
        concatenated_input += self.modality_embedding
        
        # Add positional embeddings
        concatenated_input += self.positional_embedding
        
        # Apply sequence mask if given as parameter
        if sequence_mask is not None:

            # Transformer decoder output
            transformer_output = self.transformer_decoder(
                tgt=concatenated_input,
                memory=concatenated_input,
                tgt_mask=sequence_mask,
                memory_mask=sequence_mask
            )
            
            # Remove batch dimension
            transformer_output = transformer_output.squeeze(0)
            
            # Apply output embedding to get probability map
            probability_map = self.output_embedding(transformer_output)
            
            return probability_map
        
        else:
            pass

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience=5):
    # use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        running_loss = 0.0

        # train with progress bar
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):
            inputs = [inp.to(device) for inp in inputs]
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs[0].size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")

        # Evaluate on validation set
        val_loss = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

        # Check for early stopping with patience
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping!")
                break

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = [inp.to(device) for inp in inputs]
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs[0].size(0)
    return total_loss / len(data_loader.dataset)

# Example usage:
# train_dataset = our train dataset
# val_dataset = our validation dataset
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Instantiate the model
# model = Decoder4M(hidden_dim, num_decoder_layers, num_attention_heads, window_size)

# Define loss and optimizer
# criterion = likely_loss_function.MaskedMSELoss()

# would have to import optim
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
# train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience=5)

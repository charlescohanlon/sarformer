import torch
import torch.nn as nn
from tqdm import tqdm
from swinv2_encoder import SwinTransformerV2
from bert_encoder import BertEncoder
from tabular_encoder import TabularEncoder
from decoder import Decoder


class SARFormer(nn.Module):
    def __init__(self):
        self.swin_encoder = SwinTransformerV2(img_size=512, patch_size=4, in_chans=4)
        self.bert_encoder = BertEncoder()
        self.tabular_encoder = TabularEncoder()
        self.decoder = Decoder()

    def forward(self, x):
        swin_embed = self.swin_encoder(x)  # TODO: fill this in
        bert_embed = self.bert_encoder(x)
        tab_embed = self.tabular_encoder(x)

        # Concat along sequence dimension
        cat_embed = torch.cat((swin_embed, bert_embed, tab_embed), dim=1)

        return self.decoder(cat_embed)


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


def train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs, patience=5
):
    # use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        running_loss = 0.0

        # train with progress bar
        for inputs, targets in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"
        ):
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

import torch
import torch.nn as nn
from tqdm import tqdm


class SARFormer(nn.Module):
    # Create random tensors to represent latent feature representations from swin_v2 and ALBERT
    # Note that dim1 is the batch size, dim2 is the sequence length (for ALBERT) and
    # window_size^2 for swin_v2, and 768 is the hidden dimension size for both models.
    tensor_swin = torch.randn(64, 100, 768)  # assuming batch size 64, window size 10
    tensor_albert = torch.randn(
        64, 200, 768
    )  # assuming batch size 64, sequence length 200

    # Concatenate along the second dimension (dim=1)
    concatenated_tensor = torch.cat((tensor_swin, tensor_albert), dim=1)

    print(concatenated_tensor.shape)  # Output: torch.Size([64, 300, 768])

    # We would then input the concatenated tensor into the decoder


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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sarformer import SARFormer
from tqdm import tqdm
from datasets import load_dataset
from transformers import BertTokenizer

def generate_fake_data(num_samples, text_max_seq_len):
    image_data = torch.randn(num_samples, 4, 512, 512)
    dataset = load_dataset("imdb", split="train[:{}]".format(num_samples))
    texts = dataset["text"][:num_samples]

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    text_data = []
    for text in texts:
        max_seq = max(len(text), 3000)
        text_data.append(text[:max_seq])

    image_data = torch.randn(num_samples, 4, 512, 512)
    tabular_data = torch.randn(num_samples, 10)
    return image_data, text_data, tabular_data

class CustomDataset(Dataset):
    def __init__(self, image_data, text_data, tabular_data):
        self.image_data = image_data
        self.text_data = text_data
        self.tabular_data = tabular_data

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        image = self.image_data[idx]
        text = self.text_data[idx]
        tabular = self.tabular_data[idx]
        return image, text, tabular

def collate_fn(batch):
    images, texts, tabulars = zip(*batch)
    # need to change to providing tensors to bert input instead of text
    # texts = torch.stack(torch.tensor(texts))
    images = torch.stack(images)
    tabulars = torch.stack(tabulars)
    return images, texts, tabulars

num_samples = 1000
text_max_seq_len = 1024

image_data, text_data, tabular_data = generate_fake_data(num_samples, text_max_seq_len)
train_dataset = CustomDataset(image_data, text_data, tabular_data)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

model = SARFormer(img_size=512, dim_embed=768, num_tab_features=10, text_max_seq_len=text_max_seq_len, mask_proportions={"swin": 0.1, "bert": 0.1, "tabular": 0.1})
criterion = nn.MSELoss()  # reconstruction loss for now but replace with custom loss later
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")

    for i, (image, text, tabular) in progress_bar:
        image, tabular = image.to(device), tabular.to(device)
        
        optimizer.zero_grad()
        
        reconstructed_image, reconstructed_text, reconstructed_tabular = model(image, text, tabular)
        loss = criterion(reconstructed_image, image) + criterion(reconstructed_text, text) + criterion(reconstructed_tabular, tabular)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        if i % 100 == 99:    # Print every 100 mini-batches
            avg_loss = running_loss / 100
            progress_bar.set_postfix(loss=avg_loss)
            running_loss = 0.0

print('Finished Training')

# Fake validation data for testing
val_image_data, val_text_data, val_tabular_data = generate_fake_data(200, text_max_seq_len)
validation_dataset = CustomDataset(val_image_data, val_text_data, val_tabular_data)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

model.eval()
total_loss = 0.0

with torch.no_grad():
    for image, text, tabular in tqdm(validation_loader, desc="Validation"):
        image, text, tabular = image.to(device), text.to(device), tabular.to(device)
        
        reconstructed_image, reconstructed_text, reconstructed_tabular = model(image, text, tabular)
        loss = criterion(reconstructed_image, image) + criterion(reconstructed_text, text) + criterion(reconstructed_tabular, tabular)
        total_loss += loss.item()

print(f'Average reconstruction loss on the validation set: {total_loss / len(validation_loader):.4f}')

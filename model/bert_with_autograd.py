import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label)
        }

# Just some fake movie reviews that I asked ChatGPT to write me
texts = [
    "I love this movie!", "This movie is terrible.", "The acting was amazing.", "I didn't like the plot.", "Best movie ever!",
    "Worst movie of the year.", "The plot was confusing.", "The cinematography was beautiful.", "The characters were well-developed.",
    "I couldn't stop laughing!", "It was so boring.", "The soundtrack was incredible.", "The special effects were disappointing.",
    "I was on the edge of my seat!", "I fell asleep during the movie.", "The dialogue was witty.", "The ending was unexpected.",
    "I recommend it to everyone!", "It's not worth watching.", "I cried at the end.", "The pacing was too slow."
]
labels = [1, 0, 1, 0, 1, 0, 0, 1, 1, 1,
          0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0]  # 1 for positive sentiment, 0 for negative sentiment

train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.1, random_state=42)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

max_length = 32 
train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_length)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, max_length)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)

optimizer = AdamW(model.parameters(), lr=1e-5)

epochs = 8
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Average training loss for Epoch {epoch + 1}: {avg_loss:.4f}")


    model.eval()
    total_correct = 0
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc=f"Validation Epoch {epoch + 1}"):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            total_correct += torch.sum(predictions == labels).item()

    accuracy = total_correct / len(test_dataset)
    print(f"Validation accuracy for Epoch {epoch + 1}: {accuracy:.4f}")
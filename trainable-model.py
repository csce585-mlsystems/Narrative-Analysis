from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
import torch

# Load pre-trained BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Sample dataset (replace with your own data)
sentences = [
    "I love this product!",    # Positive
    "This is terrible.",       # Negative
]
labels = [1, 0]  # 1 for positive, 0 for negative

# Tokenize the data
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

# Create labels tensor
labels_tensor = torch.tensor(labels)

# DataLoader
data = list(zip(inputs['input_ids'], inputs['attention_mask'], labels_tensor))
loader = DataLoader(data, batch_size=2)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
model.train()
num_epochs = 3
total_steps = num_epochs * len(loader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps
)

for epoch in range(num_epochs):
    for batch in loader:
        input_ids, attention_mask, labels = batch

        # Forward pass (Note: No need to unsqueeze since DataLoader provides a batch)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        
        # Calculate loss
        loss = outputs.loss
        loss.backward()

        # Optimization step
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

print("Model fine-tuning complete!")

torch.save(model.state_dict(), '585_bert_model.pth')

tokenizer.save_pretrained('./tokenizer')
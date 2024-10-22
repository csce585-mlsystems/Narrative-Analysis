from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
import torch

# Load pre-trained BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Sample dataset (replace with your own data)
sentences = [
    "I love this product!",    # Very Positive
    "Yeah. And it didn't go so well. I ended up staying way longer and spending way more money than I'd planned to. I feel so bad. My partner's pissed. And now I'm deeper debt, and I made the situation worse again. I, I can't figure out why I keep doing this to myself.",       # Very Negative
    "No. I ended up staying for, like, eight hours, and losing all my money, that I had saved in my bank account from not gambling for a month. So I'm deeper in debt, my partner knows that I relapsed, because I stayed out for so long. I'm so stupid", #Negative
    "Well, I would say I think a little bit better. I don't know if I know exactly why, but I feel a little bit better. When I woke up in the morning, I was able to get up more easily. I think that when I was just reading the paper, even the sports section, I felt like I was able to concentrate a little better", # Positive
    "Well, it's kind of like the evenings are really hard for me because even if I go out and do something, then at the end of the day, I go back and I'm there by myself. Then I'm just sort of there with my thoughts", # Negative
    "Well, I think the idea of working toward goals every week is a good idea. I think that your helping me to figure out what those things are is good. If I evaluate my thinking, which may be 100 percent, or not, or somewhere in the middle- ", # Positive
    "Last week, I hadn't really done that much that I really felt that I deserved any credit. But I tried to push myself this week and do the things we talked about, so yeah. I was better because I was able to recognize that. ", # Neutral
    "Well, all of the things, just going out for a walk is better than sitting on the couch. But I went out with my grandson again, and that was good. ", #Positive
    "I don't know", # Neutral
    "It was easier than the first time. It was a lot easier than the first time", # Very Positive
    "Yeah. I guess the worst one was like when I said that sitting on the couch is hard, so when I was sitting there, there would be times when I would think that I'm not going to get better", # Very Negative
    "This therapy and all that. So, I went out and had ice cream with my grandson. [expressing additional automatic thoughts] Well, I ought to be doing that anyway. I went for a walk. Well, I ought to be doing that anyway", # Neutral
    "Losing my job really hurt", #Negative
    "Well, I recognize then that I'm doing something, even though I have depression.", #Neutral
    "After I call him, it'll be fine. He'll be happy to hear from me. It's just dialing the number that's going to be the hard part.", #Neutral
    "Gosh. The words, I feel like, sound so harsh, but really I think of embarrassment, humiliation. I feel like I look stupid. Um, I— yeah. Those kinds of things.", #Negative
    "It does, because I really hate to be perceived as stupid", #Very Negative
    "And I can laugh. I mean, I am totally fine laughing at myself. That's not the problem. It's just, I just don't want that identity that I'm dumb", #Neutral
    "I like football and hockey, and, I enjoy eating.", #Positive
    "Kind of hard, because my brother, he—he brags, like he thinks he is just as good as—like he acts like just like an 11-year-old, and like all his friends are really kind of annoying. And, yeah. He just—gets on my nerves about 90%\ of the time", # Negative
    "I was really into enjoying him very much", #Very Positive
    "I am somewhat better and I did that.", #Positive
    "But, I mean, I think that that's the best plan I have right now", #Positive
    "I had a terrible day", #Very Negative
    "Work went badly today", #Negative
]
labels = [4, 0, 1, 3, 1, 3, 2, 3, 2, 4, 0, 2, 1, 2, 2, 1, 0, 2, 3, 1, 4, 3, 3, 0, 1]  # 4 for very positive, 3 for positive, 2 for neutral, 1 for negative, 0 for very negative

# Tokenize the data
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

# Create labels tensor
labels_tensor = torch.tensor(labels)

# DataLoader
data = list(zip(inputs['input_ids'], inputs['attention_mask'], labels_tensor))
loader = DataLoader(data, batch_size=2)

# Freeze BERT parameters
for param in model.bert.parameters():
    param.requires_grad = True # Freeze all parameters

# Define optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
model.train()
num_epochs = 10
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
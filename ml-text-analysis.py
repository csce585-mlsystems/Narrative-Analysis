import torch
from transformers import BertTokenizer, BertForSequenceClassification
from scipy.special import softmax

# Model architecture must be the same as during training
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Load the state dictionary from the saved file
model.load_state_dict(torch.load('585_bert_model.pth'))

# Put the model into evaluation mode
model.eval()

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('./tokenizer')

def analyze_sentiment(text):
    # Tokenize the input text and encode it
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Run the text through the model to get sentiment prediction logits
    with torch.no_grad():
        outputs = model(**inputs)

    # Get logits and convert to probabilities using softmax
    logits = outputs.logits
    probs = softmax(logits.numpy()[0])

    # Get the label with the highest probability
    sentiment_class = probs.argmax()
    
    # Mapping model output to the sentiment labels
    labels = ["very negative", "negative", "neutral", "positive", "very positive"]

    return labels[sentiment_class]

def main():
    while True:
        try:
            # Take input from user
            input_text = input("Enter a text (or type 'exit' to quit): ")
            if input_text.lower() == 'exit':
                break

            # Analyze the sentiment of the input text
            sentiment = analyze_sentiment(input_text)
            print(f"Sentiment: {sentiment}")

        except KeyboardInterrupt:
            print("\nExiting.")
            break

if __name__ == "__main__":
    main()

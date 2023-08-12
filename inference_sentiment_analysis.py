
from transformers import DebertaTokenizer, DebertaForSequenceClassification, get_scheduler
from torch.utils.data import DataLoader
import torch

"""## Test cases"""

# Load DeBERTa tokenizer
tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')

# Set maximum length for inputs
max_length = 256

# Define the trained model
model_testing = DebertaForSequenceClassification.from_pretrained('./Model/', num_labels=2)

# Determine the device to use for training (GPU if available, else CPU)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Move model to the chosen device
model_testing.to(device)

# user input review
review  = input("Please enter review: ")

# tokenize review and convert to PyTorch tensor
input_review_tokens_tensor = tokenizer(review, padding="max_length", truncation=True, return_tensors = 'pt')['input_ids'].to(device)
# Put the model in evaluation mode
model_testing.eval()

# Pass the review through the model for inference
with torch.no_grad():
    outputs = model_testing(input_review_tokens_tensor)

# Get the predicted class probabilities
logits = outputs.logits
predicted_probabilities = torch.softmax(logits, dim=1)

# Get the predicted class (0 for negative, 1 for positive)
_, predicted_class = torch.max(predicted_probabilities, dim=1)

# The predicted_class will be a tensor with 0 or 1.
predicted_sentiment = "positive" if predicted_class.item() == 0 else "negative"

print(f"Review: {review} <----|----> Predicted sentiment: {predicted_sentiment}")


"""
# Sentiment Analysis on IMDb movie reviews
"""

## Installing required packages
#!pip install transformers evaluate datasets

"""## Collecting data with the help of Kaggle API in Google Colab"""

# Upload your Kaggle API credentials to your Colab environment
# from google.colab import files
# files.upload()
# # Move the Kaggle API credentials to the correct directory and set permissions
# !mkdir -p ~/.kaggle
# !cp kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json

# # Download the Kaggle dataset
# !kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

# # Extract the contents of the ZIP file
# !unzip imdb-dataset-of-50k-movie-reviews.zip

"""## Importing required packages"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from transformers import DebertaTokenizer, DebertaForSequenceClassification, get_scheduler
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import classification_report
import torch
from tqdm.auto import tqdm
from datasets import *
sns.set()

"""## Loading data"""

# read the csv file
df = pd.read_csv('./Dataset/IMDB Dataset.csv')

df = df

"""## Exploratory Data Analysis"""

# display top 5 rows
df.head()

# get the info
df.info()

plt.figure(figsize=(3, 4))
# Calculate the value counts of each unique value in the 'sentiment' column
sentiment_counts = df['sentiment'].value_counts()

# Create a bar plot using Seaborn
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)

# Add labels and title
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Sentiment Counts')

# Show the plot
plt.show()

"""## Data Preparation"""

# Encode the label like 0 for positive and 1 for negative
df['sentiment'] = df.sentiment.map({"positive":0, "negative":1})

# Create PyTorch dataset from pandas dataframe
dataset = Dataset.from_pandas(df)

# Load DeBERTa tokenizer
tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')

# Set maximum length for inputs
max_length = 256

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["review"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

train_testvalid = tokenized_datasets.train_test_split(test_size=0.2)
# Split the 10% test + valid in half test, half valid
test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
# gather everyone if you want to have a single DatasetDict
ds_prepared = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'valid': test_valid['train']})

# Remove the "review" column
ds_prepared = ds_prepared.remove_columns(["review"])

# Rename the "sentiment" column to "labels"
ds_prepared = ds_prepared.rename_column("sentiment", "labels")

# Set the format to PyTorch tensors
ds_prepared.set_format("torch")

# Select a subset of the train and test datasets for checking purpose
train_dataset = ds_prepared['train'] # TODO: get all values. .select(range(500))
test_dataset = ds_prepared['test']  # TODO: get all values
valid_dataset = ds_prepared['valid']

# Create data loaders
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4)
test_dataloader = DataLoader(test_dataset, batch_size=4)
valid_dataloader = DataLoader(valid_dataset, batch_size=4)

"""## Model Training"""

# Define the DeBERTa model architecture
model = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-base', num_labels=2)

# Define optimizer and learning rate
optimizer = AdamW(model.parameters(), lr=5e-5)

# Set number of epochs for training
num_epochs = 15

# Calculate the total number of training steps
num_training_steps = num_epochs * len(train_dataloader) + num_epochs * len(valid_dataloader)

# Set learning rate scheduler
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_epochs * len(train_dataloader)
)

# Determine the device to use for training (GPU if available, else CPU)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Move model to the chosen device
model.to(device)

# Initialize the progress bar
progress_bar = tqdm(range(num_training_steps))

# Lists to store training and validation losses for plotting
train_losses = []
valid_losses = []

best_loss = float('inf')
patience_counter = 0
early_stop_patience = 2  # Set the patience threshold

# Loop through the specified number of epochs
for epoch in range(num_epochs):
    # Put the model in training mode
    model.train()

    # Lists to store validation loss per epoch
    train_loss_per_epoch = []

    # Loop through each batch in the training data loader
    for batch in train_dataloader:
        # Move the batch to the device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Compute the outputs of the model
        outputs = model(**batch)

        # Compute the loss
        loss = outputs.loss
        loss.backward()

        # Update the parameters of the optimizer
        optimizer.step()

        # Update the learning rate scheduler
        lr_scheduler.step()

        # Zero out the gradients
        optimizer.zero_grad()

        # Store training loss for plotting
        train_loss_per_epoch.append(loss.item())

        # Display training loss in the progress bar
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item())})

        # Update the progress bar
        progress_bar.update(1)

    # Calculate the average training loss for the epoch
    avg_train_loss = np.mean(train_loss_per_epoch)
    train_losses.append(avg_train_loss)

    # Validation loop after each epoch
    model.eval()  # Put the model in evaluation mode

    # Lists to store validation loss per epoch
    valid_loss_per_epoch = []

    # Loop through the validation data loader
    with torch.no_grad():
        for batch in valid_dataloader:
            # Move the batch to the device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Compute the outputs of the model
            outputs = model(**batch)

            # Compute the loss
            loss = outputs.loss

            # Store validation loss for plotting
            valid_loss_per_epoch.append(loss.item())

            # Display training loss in the progress bar
            progress_bar.set_postfix({'validation_loss': '{:.3f}'.format(loss.item())})

            # Update the progress bar
            progress_bar.update(1)

    # Calculate the average validation loss for the epoch
    avg_valid_loss = np.mean(valid_loss_per_epoch)
    valid_losses.append(avg_valid_loss)

    # save the model a the end of the epoch
    model.save_pretrained("/content/model/")

    # Check for early stopping
    if avg_valid_loss < best_loss:
        best_loss = avg_valid_loss
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= early_stop_patience:
        print(f'Early stopping triggered at epoch {epoch + 1}.')
        break

# Plot final epoch-wise training and validation loss
plt.plot(train_losses, label='Training Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

"""## Evaluate Model on Test dataset"""

# Set the model to evaluation mode
model.eval()

# Create empty arrays to store true labels and predicted labels
true_labels = []
predicted_labels = []

# Iterate over the test dataloader
for batch in test_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        # Forward pass through the model
        outputs = model(**batch)

    # Get the predicted labels
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

    # Add the batch to the true and predicted labels arrays
    true_labels.extend(batch["labels"].cpu().numpy())
    predicted_labels.extend(predictions.cpu().numpy())

# labels
class_names = ["Positive", "Negative"]
# Compute precision, recall, F1 score, and support for each class
class_report = classification_report(true_labels, predicted_labels, target_names=class_names)

# Display the classification report
print(class_report)

"""## Saving the model"""

model.save_pretrained("/Model/")
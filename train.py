import json
import numpy as np
import torch
import torch.nn as nn
import random
import string
from nltk_utils import tokenize, stem, bag_of_words
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet

# import nltk
# nltk.download('punkt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

# Load intents data
with open('intents.json', 'r') as file:
    intents = json.load(file)

# Extract input-output pairs from intents data
all_words = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

# Stem and lower each word
ignore_words = set(string.punctuation)
all_words = [stem(w) for w in all_words if w not in ignore_words]
# Sort and remove duplicates
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Print information about the data
# print(len(xy), "patterns")
# print(len(tags), "tags:", tags)
# print(len(all_words), "unique stemmed words:", all_words)

# create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label) # CrossEntropyLoss

X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # dataset[index]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Hyperparameters
input_size = len(all_words) # Model parameters
# input_size = len(X_train[0]) 
hidden_size = 16
output_size = len(tags)
num_epochs = 1500 # Training parameters
batch_size = 8
learning_rate = 0.001
# print(input_size, output_size)

# Create dataloader
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create model
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# Training loop
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'final loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
X_test = X_train
y_test = y_train

with torch.no_grad():
    X_test = torch.tensor(X_test).to(device)
    y_test = torch.tensor(y_test).to(device)

    X_test = X_test.to(device)
    y_test = y_test.to(device)

    outputs = model(X_test)
    _, predicted = torch.max(outputs, dim=1)

    predicted = predicted.cpu().numpy()
    y_test = y_test.cpu().numpy()

    accuracy = accuracy_score(y_test, predicted)
    precision = precision_score(y_test, predicted, average='weighted')
    recall = recall_score(y_test, predicted, average='weighted')
    f1 = f1_score(y_test, predicted, average='weighted')

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1:.4f}')

# Save trained model
FILE = "data.pth"
torch.save({
    'input_size': input_size,
    'hidden_size': hidden_size,
    'output_size': output_size,
    'all_words': all_words,
    'tags': tags,
    'model_state': model.state_dict()
}, FILE)
print(f'training complete. file saved to {FILE}')

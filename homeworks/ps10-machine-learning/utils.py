import re
import numpy as np

# Function to load the dataset
def load_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    labels = []
    messages = []
    for line in lines:  # Skip the header
        label, sms = line.strip().split('\t')
        labels.append(label)
        messages.append(sms)
    return labels, messages

# Function to replace punctuation in text
def replace_punctuation(text):
    """
    Replaces all punctuation in the text with spaces.
    """
    return re.sub(r'[^\w\s]', ' ', text)

# Function to clean SMS messages
def clean_data(labels, messages):
    cleaned_labels = labels  # Labels don't need cleaning
    cleaned_messages = [replace_punctuation(msg).lower() for msg in messages]
    return cleaned_labels, cleaned_messages

# Function to shuffle the dataset
def shuffle_data(labels, messages, seed=1):
    np.random.seed(seed)
    indices = np.arange(len(labels))
    np.random.shuffle(indices)
    shuffled_labels = [labels[i] for i in indices]
    shuffled_messages = [messages[i] for i in indices]
    return shuffled_labels, shuffled_messages

# Function to split the dataset into training and test sets
def split_data(labels, messages, train_ratio=0.8):
    split_index = int(len(labels) * train_ratio)
    training_labels = labels[:split_index]
    training_messages = messages[:split_index]
    test_labels = labels[split_index:]
    test_messages = messages[split_index:]
    return (training_labels, training_messages), (test_labels, test_messages)

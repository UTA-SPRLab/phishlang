import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split, KFold
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support
from concurrent.futures import ThreadPoolExecutor
import torch

def read_text(filename, label):
    directory = 'phish_samples' if label == 1 else 'benign_samples'
    try:
        return open(os.path.join(directory, filename), encoding="utf-8", errors="ignore").read()
    except:
        return ""

# Sliding window 
def chunk_text(text, tokenizer, window_size=128, stride=64):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids.squeeze()
    total_tokens = tokens.size(0)
    chunks = []
    
    for i in range(0, total_tokens, stride):
        window = tokens[i:i+window_size]
        if window.size(0) < window_size:
            break  # Ensures the window size matches the required size
        
        chunk = tokenizer.decode(window, skip_special_tokens=True)
        chunks.append(chunk)
    
    return chunks

# Check command line arguments
if len(sys.argv) < 2:
    print("Usage: script.py <test|full> [chunk]")
    sys.exit(1)

mode = sys.argv[1]
use_chunking = "chunk" in sys.argv

tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")

phish_files = [f for f in os.listdir('phish_samples') if f.endswith('.txt')]
benign_files = [f for f in os.listdir('benign_samples') if f.endswith('.txt')]

phishing = [(f, 1) for f in phish_files]
benign = [(f, 0) for f in benign_files]

data = phishing + benign
df = pd.DataFrame(data, columns=['file', 'label'])

def parallel_read(file_label):
    return read_text(*file_label)

with ThreadPoolExecutor(max_workers=8) as executor:
    texts = list(executor.map(parallel_read, zip(df['file'], df['label'])))
df['text'] = texts

# Select 100 random samples for test mode
if mode == "test":
    df = df.sample(100, random_state=42)

# If chunking is enabled, create chunks of the text
if use_chunking:
    all_chunks = []
    all_labels = []
    for _, row in df.iterrows():
        chunks = chunk_text(row['text'], tokenizer)
        all_chunks.extend(chunks)
        all_labels.extend([row['label']] * len(chunks))
    df = pd.DataFrame({'text': all_chunks, 'label': all_labels})

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

# Pre-tokenize the entire dataset once
full_dataset = Dataset.from_pandas(df[['text', 'label']])
full_dataset = full_dataset.map(tokenize_function, batched=True)

for train_index, val_index in kf.split(train_df):
    fold_train_df = train_df.iloc[train_index]
    fold_val_df = train_df.iloc[val_index]
    
    train_dataset = full_dataset.select(train_index.tolist())
    val_dataset = full_dataset.select(val_index.tolist())
    test_dataset = Dataset.from_pandas(test_df[['text', 'label']])
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    model = MobileBertForSequenceClassification.from_pretrained("google/mobilebert-uncased", num_labels=2)
    
    args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        weight_decay=0.01,
        gradient_accumulation_steps=4
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    trainer.train()
    
    predictions = trainer.predict(test_dataset)
    accuracy = (predictions.predictions.argmax(-1) == test_df['label']).mean()
    precision, recall, f1, _ = precision_recall_fscore_support(test_df['label'], predictions.predictions.argmax(-1), average='weighted')
    
    results.append((accuracy, precision, recall, f1))

for i, (accuracy, precision, recall, f1) in enumerate(results):
    print(f"Fold {i+1} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

# Save the model and tokenizer
model.save_pretrained('./model')
tokenizer.save_pretrained('./model')

print('Model saved')

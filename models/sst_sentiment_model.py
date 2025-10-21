#!/usr/bin/env python3
# sentiment_model.py

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

# --------------------------
# Dataset: SST-2
# --------------------------
print("Loading GLUE/SST-2 …")
raw_ds = load_dataset("glue", "sst2")

MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

MAX_LEN = 128

def tokenize_fn(example):
    return tokenizer(
        example["sentence"],
        truncation=True,
        max_length=MAX_LEN,
    )

print("Tokenizing …")
tok_ds = raw_ds.map(tokenize_fn, batched=True, remove_columns=[c for c in raw_ds["train"].column_names if c != "label"])
tok_ds = tok_ds.rename_column("label", "labels")
tok_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# --------------------------
# Dataloaders
# --------------------------
batch_size = 64

def collate_batch(features):
    input_ids = [f["input_ids"] for f in features]
    attention_mask = [f["attention_mask"] for f in features]
    labels = torch.tensor([f["labels"].item() for f in features], dtype=torch.long)

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    return {"input_ids": input_ids, "attention_mask": attention_mask}, labels

training_data = tok_ds["train"]
test_data = tok_ds["validation"]

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)
test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=0, collate_fn=collate_batch)

X_batch, y_batch = next(iter(test_dataloader))
print(f"input_ids batch shape: {X_batch['input_ids'].shape}")
print(f"attention_mask shape: {X_batch['attention_mask'].shape}")
print(f"labels shape: {y_batch.shape} {y_batch.dtype}")

# --------------------------
# Device (keep a simple, robust detector)
# --------------------------
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using {device} device")

# --------------------------
# Model (we keep your variable names/flow)
# --------------------------
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

epochs = 5
num_training_steps = epochs * len(train_dataloader)
warmup_steps = int(0.06 * num_training_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

# --------------------------
# Train / Test loops 
# --------------------------
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch_idx, (X, y) in enumerate(dataloader):
        X = {k: v.to(device) for k, v in X.items()}
        y = y.to(device)

        outputs = model(**X, labels=y)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        #log every 100 batches
        if batch_idx % 100 == 0:
            current = (batch_idx + 1) * X["input_ids"].size(0)
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0.0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X = {k: v.to(device) for k, v in X.items()}
            y = y.to(device)
            outputs = model(**X, labels=y)
            logits = outputs.logits

            # accumulate loss 
            test_loss += outputs.loss.item()

            # accuracy
            preds = logits.argmax(dim=1)
            correct += (preds == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# --------------------------
# Run training + save
# --------------------------
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

os.makedirs("checkpoints_sst2", exist_ok=True)
save_dir = "checkpoints_sst2/best"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"Saved model & tokenizer to {save_dir}")

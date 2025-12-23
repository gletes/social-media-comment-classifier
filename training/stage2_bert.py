import pandas as pd
import numpy as np
import pickle
import torch
import os
import sys
import math
import time
import datetime
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import Adam
from sklearn.metrics import classification_report

# --- Configuration and Paths ---
# Assuming this file is run from the 'training' folder
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
DATA_SPLIT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data_splits')

# Intermediate Data to Load from Stage 1
PROCESSED_DF_PATH = os.path.join(DATA_SPLIT_DIR, 'df_processed.pkl')
TRAIN_INDICES_PATH = os.path.join(DATA_SPLIT_DIR, 'train_indices.pkl')
TEST_INDICES_PATH = os.path.join(DATA_SPLIT_DIR, 'test_indices.pkl')

# Multi-class Stage 2 Model Outputs
BERT_MODEL_PATH = os.path.join(MODEL_DIR, 'bert_model.pkl')
BERT_MODEL_NAME = "indobenchmark/indobert-base-p2"

# Hyperparameters (Matching the user's request)
MAX_LEN = 128
EPOCHS = 5
BATCH_SIZE = 16
LR = 2e-5

# Mapping
LABEL_MAP = {
    "Provokasi": 0, "Penghinaan": 1, "Pornografi": 2, 
    "Negatif Lainnya": 3, "SARA": 4, "Pencemaran Nama Baik": 5
}
LABEL_NAME_MAP = {v: k for k, v in LABEL_MAP.items()}

# --- Utility Functions (Copied for self-sufficiency) ---
def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))

# --- PyTorch Dataset Class (Copied for self-sufficiency) ---
class IndoBERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# --- Main Training Logic ---
def run_stage2_bert_training():
    print("--- Stage 2: IndoBERT (Multi-class Classification) Training ---")

    # 1. Load Data Splits from Stage 1
    if not os.path.exists(PROCESSED_DF_PATH):
        print(f"Error: Data splits not found in {DATA_SPLIT_DIR}. Did Stage 1 run successfully?")
        return

    try:
        df_stage2 = pd.read_pickle(PROCESSED_DF_PATH)
        train_indices = pickle.load(open(TRAIN_INDICES_PATH, 'rb'))
        test_indices = pickle.load(open(TEST_INDICES_PATH, 'rb'))
    except Exception as e:
        print(f"Error loading intermediate data: {e}")
        return

    # Create temporary binary label column for filtering
    df_stage2['flag_label'] = df_stage2['Final Label'].apply(lambda x: '0' if x == 'Non-Negatif' else '1').astype(str)

    # 2. Filter Data to Negative Subset Only
    
    # Recreate train/test sets based on indices
    df_train = df_stage2.loc[train_indices]
    df_test = df_stage2.loc[test_indices]
    
    # Filter for 'Negatif' comments only (flag_label == '1')
    X_train_stage2 = df_train[df_train['flag_label'] == '1']['sastrawi']
    y_train_stage2_raw = df_train[df_train['flag_label'] == '1']['Final Label']
    
    X_test_stage2 = df_test[df_test['flag_label'] == '1']['sastrawi']
    y_test_stage2_raw = df_test[df_test['flag_label'] == '1']['Final Label']

    # 3. Encode and Prepare DataLoaders
    y_train_stage2_encoded = y_train_stage2_raw.map(LABEL_MAP)
    y_test_stage2_encoded = y_test_stage2_raw.map(LABEL_MAP)

    print(f"Training BERT on {len(X_train_stage2)} Negative samples.")
    
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    train_dataset = IndoBERTDataset(X_train_stage2, y_train_stage2_encoded, tokenizer)
    test_dataset = IndoBERTDataset(X_test_stage2, y_test_stage2_encoded, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 4. Model Setup
    model = BertForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME,
        num_labels=len(LABEL_MAP)
    )
    model.to(device)
    
    # Calculate Class Weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_stage2_encoded.dropna()),
        y=y_train_stage2_encoded.dropna()
    )
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    criterion = torch.nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = Adam(model.parameters(), lr=LR, eps=1e-8)
    
    total_steps = len(train_loader) * EPOCHS
    num_warmup_steps = math.ceil(total_steps * 0.05)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=num_warmup_steps, 
                                                num_training_steps=total_steps)

    # 5. Training Loop
    print(f"Starting BERT Training for {EPOCHS} epoch(s)...")
    for epoch_i in range(EPOCHS):
        t0 = time.time()
        model.train()
        total_train_loss = 0
        
        for step, batch in enumerate(train_loader):
            # Move data to device
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)

            model.zero_grad()        
            outputs = model(b_input_ids, attention_mask=b_input_mask)
            
            loss = criterion(outputs.logits, b_labels)
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_loader)
        training_time = format_time(time.time() - t0)
        print(f"Epoch {epoch_i+1}/{EPOCHS} | Avg Loss: {avg_train_loss:.4f} | Time: {training_time}")

    # 6. Evaluation and Saving
    model.eval()
    predictions = []
    true_labels = []
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    for batch in test_loader:
        b_input_ids = batch['input_ids'].to(device)
        b_input_mask = batch['attention_mask'].to(device)
        b_labels = batch['labels'].to(device)
        
        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask)
        
        logits = outputs.logits
        pred_flat = torch.argmax(logits, dim=1).flatten().cpu().numpy()
        labels_flat = b_labels.cpu().numpy().flatten()
        
        predictions.extend(pred_flat)
        true_labels.extend(labels_flat)

    print("\nBERT Stage 2 Classification Report (Multi-Class on Negative Subset):")
    bert_target_names = [LABEL_NAME_MAP[i] for i in range(len(LABEL_MAP))]
    print(classification_report(true_labels, predictions, target_names=bert_target_names, zero_division=0))
    
    # Save the BERT model's state dictionary
    torch.save(model.state_dict(), BERT_MODEL_PATH)
    print(f"IndoBERT Model state dict saved to {BERT_MODEL_PATH}")

    print("\nStage 2 BERT Training Complete. All artifacts saved.")

if __name__ == '__main__':
    run_stage2_bert_training()
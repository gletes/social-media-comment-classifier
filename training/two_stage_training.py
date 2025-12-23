# For Training Part - 2 Stage
# Stage 1 = Random Forest -> Binary
# Stage 2 = BERT -> Negative only to sub categories
import pandas as pd
import numpy as np
import pickle
import torch
import os
import sys
import math
import time
import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import Adam
from sklearn.metrics import classification_report

# --- Configuration and Paths ---
DATA_FILE = 'Data Scraping Thesis - Concated Data Cut.csv'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Binary Stage 1 Model Outputs
RF_MODEL_PATH = os.path.join(MODEL_DIR, 'rf_model.pkl')
TFIDF_VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')

# Multi-class Stage 2 Model Outputs
BERT_MODEL_PATH = os.path.join(MODEL_DIR, 'bert_model.pth')
BERT_MODEL_NAME = "indobenchmark/indobert-base-p2"
MAX_LEN = 128
EPOCHS = 5
BATCH_SIZE = 16
LR = 2e-5

# Mapping
LABEL_MAP = {
    "Provokasi": 0,
    "Penghinaan": 1,
    "Pornografi": 2,
    "Negatif Lainnya": 3,
    "SARA": 4,
    "Pencemaran Nama Baik": 5
}
# Map for converting index back to name
LABEL_NAME_MAP = {v: k for k, v in LABEL_MAP.items()}

# Ensure the preprocessing module is available
try:
    from src.data_preprocessing import DataPreprocessor
except ImportError:
    print("Error: Could not import DataPreprocessor. Please ensure 'data_preprocessing.py' is in the path.")
    sys.exit(1)

# --- Utility Functions ---
def format_time(elapsed):
    """Formats time duration for logging."""
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))

# --- PyTorch Dataset Class ---
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
def run_training_pipeline():
    """Executes the full data loading, preprocessing, and model training pipeline."""
    
    print(f"--- 1. Data Loading and Preprocessing from {DATA_FILE} ---")
    
    # 1. Initialization and Data Load
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: Data file '{DATA_FILE}' not found. Please place your CSV in the current directory.")
        return

    prep = DataPreprocessor()
    
    # Check for the required 'Comments' column
    if 'Comments' not in df.columns or 'Final Label' not in df.columns:
        print("Error: Data file must contain 'Comments' and 'Final Label' columns.")
        return

    # Apply full preprocessing (cleansing -> tokenization -> stopword removal -> Sastrawi)
    print("Applying full preprocessing (Cleansing -> Stopwords -> Sastrawi)...")
    df['sastrawi'] = df['Comments'].apply(prep.full_process)

    # Create the binary label column (Stage 1 - RF)
    # Negatif: '1', Non-Negatif: '0'
    df['flag_label'] = df['Final Label'].apply(prep.flag_label).astype(str)
    
    print(f"Dataset Size: {len(df)}")
    print(f"Binary Label Distribution:\n{df['flag_label'].value_counts()}")
    
    # 2. Data Splitting
    X, y = df['sastrawi'], df['flag_label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Stage 1: Random Forest (Binary Classification) Training ---
    print("\n--- 3. Stage 1: RF Model (Binary Classification) Training ---")
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    train_tfidf = vectorizer.fit_transform(X_train)
    test_tfidf = vectorizer.transform(X_test)
    
    # Random Forest Training
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(train_tfidf, y_train)
    
    # Evaluation
    pred_test_rf = rf_model.predict(test_tfidf)
    print("RF Stage 1 Classification Report (Binary):")
    print(classification_report(y_test, pred_test_rf, target_names=['Non-Negatif (0)','Negatif (1)']))
    
    # Save the RF assets
    with open(TFIDF_VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"TF-IDF Vectorizer saved to {TFIDF_VECTORIZER_PATH}")
    
    with open(RF_MODEL_PATH, 'wb') as f:
        pickle.dump(rf_model, f)
    print(f"Random Forest Model saved to {RF_MODEL_PATH}")

    # --- Stage 2: IndoBERT (Multi-class Classification) Training ---
    print("\n--- 4. Stage 2: IndoBERT (Multi-class Classification) Training ---")

    # Filter for 'Negatif' comments only (flag_label == '1') in the training set earlier used
    train_neg_indices = y_train[y_train == '1'].index
    X_train_stage2 = df.loc[train_neg_indices, 'sastrawi']
    y_train_stage2_raw = df.loc[train_neg_indices, 'Final Label']
    
    # Encode labels (Example : 'Provokasi' -> 0)
    y_train_stage2_encoded = y_train_stage2_raw.map(LABEL_MAP)
    
    # Filter for 'Negatif' comments in the testing set for validation
    test_neg_indices = y_test[y_test == '1'].index
    X_test_stage2 = df.loc[test_neg_indices, 'sastrawi']
    y_test_stage2_raw = df.loc[test_neg_indices, 'Final Label']
    y_test_stage2_encoded = y_test_stage2_raw.map(LABEL_MAP)

    print(f"Training BERT on {len(X_train_stage2)} Negative samples.")
    
    # Prepare BERT components
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    train_dataset = IndoBERTDataset(X_train_stage2, y_train_stage2_encoded, tokenizer)
    test_dataset = IndoBERTDataset(X_test_stage2, y_test_stage2_encoded, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Model and set up optimization
    model = BertForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME,
        num_labels=len(LABEL_MAP)
    )
    model.to(device)
    
    # Calculate Class Weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_stage2_encoded.dropna()), # Use dropna in case of missing labels
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

    # --- Training Loop ---
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

    # Evaluation (BERT Stage 2)
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

    print("\n--- Training Pipeline Complete. Artifacts saved in the 'models/' directory. ---")


if __name__ == '__main__':
    run_training_pipeline()
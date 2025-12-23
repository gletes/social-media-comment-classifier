import pandas as pd
import numpy as np
import pickle
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Ensure the 'src' directory is in the path to find data_preprocessing.py
# Assuming this file is run from the project root or the 'training' folder
# If running from 'training', this path needs to go up one level to find 'src'
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 

try:
    from src.data_preprocessing import DataPreprocessor
except ImportError:
    print("Error: Could not import DataPreprocessor. Check the path in train_stage1_rf.py.")
    sys.exit(1)

# --- Configuration and Paths ---
DATA_FILE = 'Data Scraping Thesis - Concated Data Cut.csv'
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models') # models/ at project root
DATA_SPLIT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data_splits') # New folder for intermediate data

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_SPLIT_DIR, exist_ok=True)

# Binary Stage 1 Model Outputs
RF_MODEL_PATH = os.path.join(MODEL_DIR, 'rf_model.pkl')
TFIDF_VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_sastrawi.pkl')

# Intermediate Data for Stage 2
PROCESSED_DF_PATH = os.path.join(DATA_SPLIT_DIR, 'df_processed.pkl')
TRAIN_INDICES_PATH = os.path.join(DATA_SPLIT_DIR, 'train_indices.pkl')
TEST_INDICES_PATH = os.path.join(DATA_SPLIT_DIR, 'test_indices.pkl')

def run_stage1_rf_training():
    print("--- Stage 1: Random Forest (Binary Classification) Training ---")
    
    # 1. Initialization and Data Load
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: Data file '{DATA_FILE}' not found. Please place your CSV in the project root directory.")
        return

    prep = DataPreprocessor()
    
    # 2. Preprocessing
    print("Applying full preprocessing...")
    df['sastrawi'] = df['Comments'].apply(prep.full_process)

    # Create the binary label column (Negatif: '1', Non-Negatif: '0')
    df['flag_label'] = df['Final Label'].apply(prep.flag_label).astype(str)
    
    # 3. Data Splitting
    X, y = df['sastrawi'], df['flag_label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save the necessary data structures for Stage 2
    df_stage2 = df[['sastrawi', 'Final Label']]
    
    print("Saving preprocessed data and indices for Stage 2...")
    df_stage2.to_pickle(PROCESSED_DF_PATH)
    pickle.dump(X_train.index.tolist(), open(TRAIN_INDICES_PATH, 'wb'))
    pickle.dump(X_test.index.tolist(), open(TEST_INDICES_PATH, 'wb'))
    print(f"Intermediate data saved to {DATA_SPLIT_DIR}/")

    # 4. TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    train_tfidf = vectorizer.fit_transform(X_train)
    test_tfidf = vectorizer.transform(X_test)
    
    # 5. Random Forest Training
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(train_tfidf, y_train)
    
    # 6. Evaluation
    pred_test_rf = rf_model.predict(test_tfidf)
    print("\nRF Stage 1 Classification Report (Binary):")
    print(classification_report(y_test, pred_test_rf, target_names=['Non-Negatif (0)','Negatif (1)']))
    
    # 7. Save RF assets
    with open(TFIDF_VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"TF-IDF Vectorizer saved to {TFIDF_VECTORIZER_PATH}")
    
    with open(RF_MODEL_PATH, 'wb') as f:
        pickle.dump(rf_model, f)
    print(f"Random Forest Model saved to {RF_MODEL_PATH}")

    print("\nStage 1 RF Training Complete. Ready for Stage 2 BERT training.")

if __name__ == '__main__':
    run_stage1_rf_training()
# For Model Pipeline Part
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import sys

# Adjust based on actual file location (Just in case if it reads onto the `src/` folder or the other)
try:
    from data_preprocessing import DataPreprocessor 
except ImportError:
    # Fallback for deployment environments if structure is different
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from data_preprocessing import DataPreprocessor

class TwoStageModelPipeline:
    """
    Handles the two-stage text classification pipeline:
    Stage 1: RF (Binary: Non-Negatif/Negatif)
    Stage 2: BERT (Multi-class on Negatif Ones Only)
    """
    def __init__(self, rf_model_path, tfidf_path, bert_model_path, label_map):
        self.prep = DataPreprocessor()
        self.device = torch.device("cpu")
        
        # --- Stage 1: TF-DF then Random Forest ---
        # Load both the RF model & TF-IDF vectorizer
        print(f"Loading RF Model from: {rf_model_path}")
        with open(rf_model_path, "rb") as f:
            self.rf_model = pickle.load(f)
            
        print(f"Loading TF-IDF Vectorizer from: {tfidf_path}")
        with open(tfidf_path, "rb") as f:
            self.tfidf_vectorizer = pickle.load(f)
        
        # --- Stage 2: BERT ---
        print(f"Loading BERT Tokenizer and Model from: {bert_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
        
        self.bert_base_name = "indobenchmark/indobert-base-p2"
        self.bert_model = AutoModelForSequenceClassification.from_pretrained(
            self.bert_base_name, 
            num_labels=len(label_map)
        )

        self.bert_model.to(self.device)
        self.bert_model.eval()

        self.label_map = label_map
    
    def stage2_bert_predict(self, text, max_len=128):
        """Predicts the specific negative class using the BERT model."""

        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding='max_length',
            max_length=max_len
        )
        
        input_ids = enc['input_ids'].to(self.device)
        attention_mask = enc['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
            
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
        
        # Ensure the lookup is correct
        return self.label_map.get(pred, "Unknown Class") 
    
    def predict(self, text):
        # 1. Preprocess the text to get the Sastrawi version
        stemmed_text = self.prep.full_process(text)

        # --- Stage 1: Random Forest (Negatif / Non-Negatif) ---
        # TF-IDF -> Random Forest
        rf_input = self.tfidf_vectorizer.transform([stemmed_text])
        
        # Result = 0 or 1
        pred_rf_str = self.rf_model.predict(rf_input)[0] 

        if pred_rf_str == '0':
            return "Non-Negatif"
        else:
            # --- Stage 2: BERT (Negatif to Sub Categories) ---
            return self.stage2_bert_predict(stemmed_text)

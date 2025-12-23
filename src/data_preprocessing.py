import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pandas as pd

# --- FIX: Ensure NLTK resources are downloaded and available immediately ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
# --------------------------------------------------------------------------

# Function defined outside the class
def remove_punctuation_but_keep_emoticons(text):
    """
    Removes characters other than letters, numbers, spaces, and emoticons,
    based on the Unicode ranges.
    """
    return re.sub(
        r'[^\w\s'
        r'\U0001F600-\U0001F64F'
        r'\U0001F300-\U0001F5FF'
        r'\U0001F680-\U0001F6FF'
        r'\U0001F700-\U0001F77F'
        r'\U0001F780-\U0001F7FF'
        r'\U0001F800-\U0001F8FF'
        r'\U0001F900-\U0001F9FF'
        r'\U0001FA00-\U0001FA6F'
        r'\U0001FA70-\U0001FAFF'
        r'\U00002702-\U000027B0'
        r'\U0001F004-\U0001F0CF'
        r']',
        ' ',
        text
    )

class DataPreprocessor:
    """
    Encapsulates all the preprocessing steps required for the two-stage model.
    Initializes Sastrawi Stemmer and NLTK stopwords.
    """
    def __init__(self):
        # Initialize Stopwords and Sastrawi Stemmer
        self.stopwords = set(stopwords.words("indonesian"))
        self.stemmer = StemmerFactory().create_stemmer() 
    
    def cleaning(self, text):
        """Applies regex cleaning: lowercase, remove numbers, URLs, mentions, and punctuation."""
        if pd.isna(text):
            return ""
            
        text = str(text).lower()
        text = re.sub(r"\d+", "", text)  # Hapus angka
        text = re.sub(r"http[s]?://\S+|www\.\S+", " ", text)  # Hapus URL
        text = re.sub(r"@\S+", " ", text)  # Hapus mention
        text = remove_punctuation_but_keep_emoticons(text)  # Hapus simbol kecuali emoticon
        text = re.sub(r"\s+", " ", text).strip()  # Rapikan spasi
        return text
    
    def tokenize(self, text):
        """Tokenizes the cleaned text."""
        return word_tokenize(text)
    
    def stopword_removal(self, tokens): 
        """Removes common Indonesian stopwords."""
        return [t for t in tokens if t not in self.stopwords and t.strip() != ""]
    
    def lemmatize(self, text): 
        """Applies Sastrawi stemming/lemmatization."""
        return self.stemmer.stem(text)
    
    def full_process(self, text):
        """
        Runs the complete pipeline for a single piece of text.
        """
        clean_data = self.cleaning(text)
        tokenize_data = self.tokenize(clean_data)
        stopword_data = self.stopword_removal(tokenize_data)
        
        rejoined_data = " ".join(stopword_data)
        
        lemmatize_data = self.lemmatize(rejoined_data)
        return lemmatize_data
    
    def flag_label(self, final_label):
        """Converts the final label to the binary flag for Stage 1 (RF)."""
        return '0' if final_label == "Non-Negatif" else '1'

        

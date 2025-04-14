import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

class TextProcessor:
    """
    A class for basic text preprocessing operations.
    This class provides methods for cleaning and normalizing text data.
    """
    
    def __init__(self):
        """
        Initialize the TextProcessor with necessary NLTK resources.
        """
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """
        Clean the input text by:
        1. Converting to lowercase
        2. Removing special characters and numbers
        3. Removing extra whitespace
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text):
        """
        Tokenize the input text into words.
        
        Args:
            text (str): Input text to tokenize
            
        Returns:
            list: List of tokens
        """
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        """
        Remove stopwords from the token list.
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Tokens with stopwords removed
        """
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize(self, tokens):
        """
        Lemmatize the tokens to their base form.
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def process_text(self, text):
        """
        Process the text through the complete pipeline:
        1. Clean text
        2. Tokenize
        3. Remove stopwords
        4. Lemmatize
        
        Args:
            text (str): Input text to process
            
        Returns:
            list: Processed tokens
        """
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(cleaned_text)
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Lemmatize
        tokens = self.lemmatize(tokens)
        
        return tokens

# Example usage
if __name__ == "__main__":
    # Initialize the processor
    processor = TextProcessor()
    
    # Example text
    sample_text = """
    Natural Language Processing (NLP) is a subfield of artificial intelligence 
    that focuses on the interaction between computers and human language. 
    It involves the development of algorithms and models that enable computers 
    to understand, interpret, and generate human language in a valuable way.
    """
    
    # Process the text
    processed_tokens = processor.process_text(sample_text)
    
    print("Original text:", sample_text)
    print("\nProcessed tokens:", processed_tokens) 
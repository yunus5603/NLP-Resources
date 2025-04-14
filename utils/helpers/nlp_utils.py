import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from foundation.text_preprocessing.text_processor import TextProcessor
from foundation.word_embeddings.word_embeddings import WordEmbeddings
from utils.visualization.embedding_visualizer import EmbeddingVisualizer

class NLPUtils:
    """
    A utility class that combines text preprocessing and word embeddings functionality.
    """
    
    def __init__(self):
        """
        Initialize the NLPUtils class.
        """
        self.text_processor = TextProcessor()
        self.word_embeddings = None
        self.visualizer = EmbeddingVisualizer()
    
    def preprocess_text(self, text):
        """
        Preprocess text using the TextProcessor.
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of processed tokens
        """
        return self.text_processor.process_text(text)
    
    def preprocess_sentences(self, sentences):
        """
        Preprocess a list of sentences.
        
        Args:
            sentences (list): List of sentences
            
        Returns:
            list: List of processed sentences (each sentence is a list of tokens)
        """
        processed_sentences = []
        for sentence in sentences:
            tokens = self.text_processor.process_text(sentence)
            if tokens:  # Only add non-empty sentences
                processed_sentences.append(tokens)
        return processed_sentences
    
    def train_embeddings(self, sentences, vector_size=100, window=5, min_count=1, workers=4):
        """
        Train word embeddings on the provided sentences.
        
        Args:
            sentences (list): List of tokenized sentences
            vector_size (int): Dimension of the embedding vectors
            window (int): Context window size
            min_count (int): Minimum word count to include in vocabulary
            workers (int): Number of worker threads
            
        Returns:
            Word2Vec: Trained Word2Vec model
        """
        self.word_embeddings = WordEmbeddings()
        return self.word_embeddings.train_word2vec(
            sentences, 
            vector_size=vector_size, 
            window=window, 
            min_count=min_count, 
            workers=workers
        )
    
    def load_embeddings(self, model_path):
        """
        Load pre-trained word embeddings.
        
        Args:
            model_path (str): Path to the model file
            
        Returns:
            Word2Vec: Loaded Word2Vec model
        """
        self.word_embeddings = WordEmbeddings()
        return self.word_embeddings.load_model(model_path)
    
    def get_word_vector(self, word):
        """
        Get the embedding vector for a word.
        
        Args:
            word (str): Word to get embedding for
            
        Returns:
            numpy.ndarray: Embedding vector
        """
        if self.word_embeddings is None:
            raise ValueError("No embeddings loaded or trained")
        
        return self.word_embeddings.get_vector(word)
    
    def find_similar_words(self, word, n=10):
        """
        Find words similar to the given word.
        
        Args:
            word (str): Word to find similar words for
            n (int): Number of similar words to return
            
        Returns:
            list: List of (word, similarity) tuples
        """
        if self.word_embeddings is None:
            raise ValueError("No embeddings loaded or trained")
        
        return self.word_embeddings.find_similar_words(word, n=n)
    
    def solve_analogy(self, word1, word2, word3, n=5):
        """
        Solve word analogies: word1 : word2 :: word3 : ?
        
        Args:
            word1 (str): First word in the analogy
            word2 (str): Second word in the analogy
            word3 (str): Third word in the analogy
            n (int): Number of results to return
            
        Returns:
            list: List of (word, score) tuples
        """
        if self.word_embeddings is None:
            raise ValueError("No embeddings loaded or trained")
        
        return self.word_embeddings.analogy(word1, word2, word3, n=n)
    
    def visualize_embeddings(self, words, method='tsne', save_path=None):
        """
        Visualize word embeddings in 2D space.
        
        Args:
            words (list): List of words to visualize
            method (str): Dimensionality reduction method ('tsne' or 'pca')
            save_path (str): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        if self.word_embeddings is None:
            raise ValueError("No embeddings loaded or trained")
        
        # Get embeddings for the words
        embeddings = {}
        for word in words:
            vector = self.word_embeddings.get_vector(word)
            if vector is not None:
                embeddings[word] = vector
        
        return self.visualizer.plot_embedding_space(
            embeddings, 
            words, 
            method=method, 
            save_path=save_path
        )
    
    def visualize_similarity(self, words, save_path=None):
        """
        Visualize similarity matrix between words.
        
        Args:
            words (list): List of words to compare
            save_path (str): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        if self.word_embeddings is None:
            raise ValueError("No embeddings loaded or trained")
        
        # Get embeddings for the words
        embeddings = {}
        for word in words:
            vector = self.word_embeddings.get_vector(word)
            if vector is not None:
                embeddings[word] = vector
        
        return self.visualizer.plot_similarity_matrix(
            embeddings, 
            words, 
            save_path=save_path
        )
    
    def visualize_analogy(self, word1, word2, word3, word4=None, save_path=None):
        """
        Visualize vectors for word analogy.
        
        Args:
            word1 (str): First word in the analogy
            word2 (str): Second word in the analogy
            word3 (str): Third word in the analogy
            word4 (str, optional): Fourth word in the analogy (if known)
            save_path (str): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        if self.word_embeddings is None:
            raise ValueError("No embeddings loaded or trained")
        
        # Get embeddings for the words
        embeddings = {}
        for word in [word1, word2, word3]:
            vector = self.word_embeddings.get_vector(word)
            if vector is not None:
                embeddings[word] = vector
        
        if word4:
            vector = self.word_embeddings.get_vector(word4)
            if vector is not None:
                embeddings[word4] = vector
        
        return self.visualizer.plot_analogy_vectors(
            embeddings, 
            word1, 
            word2, 
            word3, 
            word4, 
            save_path=save_path
        )
    
    def save_embeddings(self, path):
        """
        Save the current embeddings model.
        
        Args:
            path (str): Path to save the model
            
        Returns:
            bool: True if successful
        """
        if self.word_embeddings is None:
            raise ValueError("No embeddings to save")
        
        return self.word_embeddings.save_model(path)

# Example usage
if __name__ == "__main__":
    # Initialize NLP utils
    nlp_utils = NLPUtils()
    
    # Example text
    text = "Natural Language Processing is amazing. Machine learning is powerful."
    
    # Preprocess text
    processed_tokens = nlp_utils.preprocess_text(text)
    print("Processed tokens:", processed_tokens)
    
    # Example sentences for training
    sentences = [
        "Natural Language Processing is amazing",
        "Machine learning is powerful",
        "Deep learning is revolutionary",
        "Neural networks are incredible",
        "Artificial intelligence is the future"
    ]
    
    # Preprocess sentences
    processed_sentences = nlp_utils.preprocess_sentences(sentences)
    print("\nProcessed sentences:")
    for i, sent in enumerate(processed_sentences):
        print(f"{i+1}. {sent}")
    
    # Train embeddings
    print("\nTraining embeddings...")
    nlp_utils.train_embeddings(processed_sentences, vector_size=50, window=3, min_count=1)
    
    # Find similar words
    test_words = ['learning', 'neural', 'processing']
    for word in test_words:
        print(f"\nWords similar to '{word}':")
        similar_words = nlp_utils.find_similar_words(word, n=3)
        for similar_word, score in similar_words:
            print(f"  {similar_word}: {score:.4f}")
    
    # Solve analogies
    analogies = [
        ('language', 'processing', 'text'),
        ('machine', 'learning', 'deep')
    ]
    
    print("\nWord Analogies:")
    for word1, word2, word3 in analogies:
        print(f"\n{word1} : {word2} :: {word3} : ?")
        results = nlp_utils.solve_analogy(word1, word2, word3, n=3)
        for word, score in results:
            print(f"  {word}: {score:.4f}")
    
    # Visualize embeddings
    words_to_visualize = ['natural', 'language', 'processing', 'machine', 'learning', 'deep', 'neural', 'networks']
    fig = nlp_utils.visualize_embeddings(words_to_visualize)
    plt.show() 
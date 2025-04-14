import numpy as np
import gensim
from gensim.models import Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import get_tmpfile
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

class WordEmbeddings:
    """
    A class for working with word embeddings.
    Supports both training custom embeddings and loading pre-trained models.
    """
    
    def __init__(self, embedding_type='word2vec', model_path=None):
        """
        Initialize the WordEmbeddings class.
        
        Args:
            embedding_type (str): Type of embedding to use ('word2vec', 'glove')
            model_path (str): Path to pre-trained model (if using pre-trained)
        """
        self.embedding_type = embedding_type
        self.model = None
        self.model_path = model_path
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def train_word2vec(self, sentences, vector_size=100, window=5, min_count=1, workers=4):
        """
        Train a Word2Vec model on the provided sentences.
        
        Args:
            sentences (list): List of tokenized sentences
            vector_size (int): Dimension of the embedding vectors
            window (int): Context window size
            min_count (int): Minimum word count to include in vocabulary
            workers (int): Number of worker threads
            
        Returns:
            Word2Vec: Trained Word2Vec model
        """
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers
        )
        return self.model
    
    def load_glove(self, glove_path, word2vec_output_path):
        """
        Convert GloVe embeddings to Word2Vec format and load them.
        
        Args:
            glove_path (str): Path to GloVe embeddings file
            word2vec_output_path (str): Path to save converted Word2Vec model
            
        Returns:
            Word2Vec: Loaded Word2Vec model
        """
        # Convert GloVe to Word2Vec format
        glove2word2vec(glove_path, word2vec_output_path)
        
        # Load the converted model
        self.model = Word2Vec.load(word2vec_output_path)
        return self.model
    
    def load_model(self, model_path):
        """
        Load a pre-trained Word2Vec model.
        
        Args:
            model_path (str): Path to the model file
            
        Returns:
            Word2Vec: Loaded Word2Vec model
        """
        self.model = Word2Vec.load(model_path)
        return self.model
    
    def get_vector(self, word):
        """
        Get the embedding vector for a word.
        
        Args:
            word (str): Word to get embedding for
            
        Returns:
            numpy.ndarray: Embedding vector
        """
        if self.model is None:
            raise ValueError("No model loaded or trained")
        
        try:
            return self.model.wv[word]
        except KeyError:
            return None
    
    def find_similar_words(self, word, n=10):
        """
        Find words similar to the given word.
        
        Args:
            word (str): Word to find similar words for
            n (int): Number of similar words to return
            
        Returns:
            list: List of (word, similarity) tuples
        """
        if self.model is None:
            raise ValueError("No model loaded or trained")
        
        try:
            return self.model.wv.most_similar(word, topn=n)
        except KeyError:
            return []
    
    def visualize_embeddings(self, words, method='tsne'):
        """
        Visualize word embeddings in 2D space.
        
        Args:
            words (list): List of words to visualize
            method (str): Dimensionality reduction method ('tsne' or 'pca')
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        if self.model is None:
            raise ValueError("No model loaded or trained")
        
        # Get vectors for the words
        vectors = []
        valid_words = []
        
        for word in words:
            try:
                vector = self.model.wv[word]
                vectors.append(vector)
                valid_words.append(word)
            except KeyError:
                continue
        
        if not vectors:
            raise ValueError("No valid words found in the model")
        
        vectors = np.array(vectors)
        
        # Apply dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        else:  # PCA
            reducer = PCA(n_components=2)
        
        reduced_vectors = reducer.fit_transform(vectors)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot points
        ax.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], alpha=0.7)
        
        # Add labels
        for i, word in enumerate(valid_words):
            ax.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]))
        
        ax.set_title(f'Word Embeddings Visualization ({method.upper()})')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        
        plt.tight_layout()
        return fig
    
    def save_model(self, path):
        """
        Save the current model to a file.
        
        Args:
            path (str): Path to save the model
            
        Returns:
            bool: True if successful
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(path)
        return True
    
    def analogy(self, word1, word2, word3, n=5):
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
        if self.model is None:
            raise ValueError("No model loaded or trained")
        
        try:
            return self.model.wv.most_similar(positive=[word2, word3], negative=[word1], topn=n)
        except KeyError:
            return []

# Example usage
if __name__ == "__main__":
    # Example sentences for training
    sentences = [
        ['natural', 'language', 'processing', 'is', 'amazing'],
        ['machine', 'learning', 'is', 'powerful'],
        ['deep', 'learning', 'is', 'revolutionary'],
        ['neural', 'networks', 'are', 'incredible'],
        ['artificial', 'intelligence', 'is', 'the', 'future'],
        ['nlp', 'is', 'part', 'of', 'ai'],
        ['word', 'embeddings', 'are', 'useful'],
        ['semantic', 'analysis', 'is', 'important'],
        ['text', 'classification', 'is', 'a', 'task'],
        ['sentiment', 'analysis', 'is', 'popular']
    ]
    
    # Initialize and train a Word2Vec model
    embeddings = WordEmbeddings()
    model = embeddings.train_word2vec(sentences, vector_size=50, window=3, min_count=1)
    
    # Find similar words
    print("Words similar to 'learning':")
    print(embeddings.find_similar_words('learning'))
    
    # Solve an analogy
    print("\nAnalogy: 'king' : 'queen' :: 'man' : ?")
    print(embeddings.analogy('king', 'queen', 'man'))
    
    # Visualize some embeddings
    words_to_visualize = ['natural', 'language', 'processing', 'machine', 'learning', 'deep', 'neural', 'networks']
    fig = embeddings.visualize_embeddings(words_to_visualize)
    plt.show() 
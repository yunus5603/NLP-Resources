import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingVisualizer:
    """
    A utility class for visualizing word embeddings.
    """
    
    @staticmethod
    def plot_embedding_space(embeddings, words, method='tsne', figsize=(12, 10), 
                            title=None, save_path=None):
        """
        Plot word embeddings in 2D space.
        
        Args:
            embeddings (dict): Dictionary mapping words to their embedding vectors
            words (list): List of words to visualize
            method (str): Dimensionality reduction method ('tsne' or 'pca')
            figsize (tuple): Figure size
            title (str): Plot title
            save_path (str): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        # Get vectors for the words
        vectors = []
        valid_words = []
        
        for word in words:
            if word in embeddings:
                vectors.append(embeddings[word])
                valid_words.append(word)
        
        if not vectors:
            raise ValueError("No valid words found in the embeddings")
        
        vectors = np.array(vectors)
        
        # Apply dimensionality reduction
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
            method_name = 't-SNE'
        else:  # PCA
            reducer = PCA(n_components=2)
            method_name = 'PCA'
        
        reduced_vectors = reducer.fit_transform(vectors)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot points
        scatter = ax.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], 
                            alpha=0.7, s=100)
        
        # Add labels
        for i, word in enumerate(valid_words):
            ax.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold')
        
        # Set title and labels
        if title is None:
            title = f'Word Embeddings Visualization ({method_name})'
        
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Dimension 1', fontsize=12)
        ax.set_ylabel('Dimension 2', fontsize=12)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_similarity_matrix(embeddings, words, figsize=(10, 8), 
                              title=None, save_path=None):
        """
        Plot similarity matrix between words.
        
        Args:
            embeddings (dict): Dictionary mapping words to their embedding vectors
            words (list): List of words to compare
            figsize (tuple): Figure size
            title (str): Plot title
            save_path (str): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        # Get vectors for the words
        vectors = []
        valid_words = []
        
        for word in words:
            if word in embeddings:
                vectors.append(embeddings[word])
                valid_words.append(word)
        
        if not vectors:
            raise ValueError("No valid words found in the embeddings")
        
        vectors = np.array(vectors)
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(vectors)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(similarity_matrix, annot=True, cmap='viridis', 
                    xticklabels=valid_words, yticklabels=valid_words,
                    vmin=0, vmax=1, ax=ax)
        
        # Set title
        if title is None:
            title = 'Word Similarity Matrix'
        
        ax.set_title(title, fontsize=16)
        
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_analogy_vectors(embeddings, word1, word2, word3, word4=None, 
                            figsize=(12, 10), title=None, save_path=None):
        """
        Plot vectors for word analogy.
        
        Args:
            embeddings (dict): Dictionary mapping words to their embedding vectors
            word1 (str): First word in the analogy
            word2 (str): Second word in the analogy
            word3 (str): Third word in the analogy
            word4 (str, optional): Fourth word in the analogy (if known)
            figsize (tuple): Figure size
            title (str): Plot title
            save_path (str): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        # Check if all words are in embeddings
        for word in [word1, word2, word3]:
            if word not in embeddings:
                raise ValueError(f"Word '{word}' not found in embeddings")
        
        if word4 and word4 not in embeddings:
            raise ValueError(f"Word '{word4}' not found in embeddings")
        
        # Get vectors
        v1 = embeddings[word1]
        v2 = embeddings[word2]
        v3 = embeddings[word3]
        
        # Calculate analogy vector
        analogy_vector = v2 - v1 + v3
        
        # If word4 is provided, get its vector
        v4 = embeddings[word4] if word4 else None
        
        # Create visualization
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot vectors
        ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, 
                  color='blue', label=word1)
        ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, 
                  color='green', label=word2)
        ax.quiver(0, 0, v3[0], v3[1], angles='xy', scale_units='xy', scale=1, 
                  color='red', label=word3)
        ax.quiver(0, 0, analogy_vector[0], analogy_vector[1], angles='xy', 
                  scale_units='xy', scale=1, color='purple', label='Analogy Vector')
        
        if v4 is not None:
            ax.quiver(0, 0, v4[0], v4[1], angles='xy', scale_units='xy', scale=1, 
                      color='orange', label=word4)
        
        # Set title and labels
        if title is None:
            title = f'Word Analogy: {word1} : {word2} :: {word3} : ?'
        
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Dimension 1', fontsize=12)
        ax.set_ylabel('Dimension 2', fontsize=12)
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

# Example usage
if __name__ == "__main__":
    # Sample embeddings (2D for visualization)
    sample_embeddings = {
        'king': np.array([1, 2]),
        'queen': np.array([1, 3]),
        'man': np.array([0, 1]),
        'woman': np.array([0, 2]),
        'boy': np.array([0.5, 1.5]),
        'girl': np.array([0.5, 2.5])
    }
    
    # Plot embedding space
    visualizer = EmbeddingVisualizer()
    fig1 = visualizer.plot_embedding_space(
        sample_embeddings, 
        ['king', 'queen', 'man', 'woman', 'boy', 'girl'],
        method='pca',
        title='Sample Word Embeddings'
    )
    
    # Plot similarity matrix
    fig2 = visualizer.plot_similarity_matrix(
        sample_embeddings,
        ['king', 'queen', 'man', 'woman', 'boy', 'girl'],
        title='Sample Word Similarities'
    )
    
    # Plot analogy vectors
    fig3 = visualizer.plot_analogy_vectors(
        sample_embeddings,
        'king', 'queen', 'man', 'woman',
        title='Word Analogy: king : queen :: man : woman'
    )
    
    plt.show() 
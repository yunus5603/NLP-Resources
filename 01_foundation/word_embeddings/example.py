import os
import numpy as np
import matplotlib.pyplot as plt
import sys

from word_embeddings import WordEmbeddings
from .text_preprocessing.text_processor import TextProcessor

def load_sample_text(file_path):
    """
    Load sample text from a file.
    
    Args:
        file_path (str): Path to the text file
        
    Returns:
        str: Text content
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def preprocess_text(text, processor):
    """
    Preprocess text using the TextProcessor.
    
    Args:
        text (str): Input text
        processor (TextProcessor): Text processor instance
        
    Returns:
        list: List of processed sentences
    """
    # Split text into sentences (simple approach)
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    # Process each sentence
    processed_sentences = []
    for sentence in sentences:
        tokens = processor.process_text(sentence)
        if tokens:  # Only add non-empty sentences
            processed_sentences.append(tokens)
    
    return processed_sentences

def main():
    """
    Main function to demonstrate word embeddings.
    """
    # Initialize text processor
    processor = TextProcessor()
    
    # Create sample text if file doesn't exist
    sample_file = 'sample_text.txt'
    if not os.path.exists(sample_file):
        sample_text = """
        Natural Language Processing (NLP) is a subfield of artificial intelligence that focuses on the interaction between computers and human language.
        It involves the development of algorithms and models that enable computers to understand, interpret, and generate human language in a valuable way.
        Machine learning is a key component of NLP, providing the ability to learn patterns from data.
        Deep learning, particularly neural networks, has revolutionized NLP in recent years.
        Word embeddings are a fundamental technique in NLP, representing words as dense vectors in a continuous space.
        These embeddings capture semantic relationships between words and are used in many NLP tasks.
        Text classification is a common NLP task that involves categorizing text into predefined classes.
        Sentiment analysis is a specific type of text classification that determines the emotional tone of text.
        Named Entity Recognition (NER) identifies and classifies named entities in text, such as people, organizations, and locations.
        Machine translation is another important NLP task that translates text from one language to another.
        Question answering systems use NLP to find answers to questions posed in natural language.
        Text summarization creates concise summaries of longer texts while preserving the most important information.
        """
        
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write(sample_text)
    
    # Load and preprocess text
    text = load_sample_text(sample_file)
    processed_sentences = preprocess_text(text, processor)
    
    print(f"Processed {len(processed_sentences)} sentences")
    
    # Initialize word embeddings
    embeddings = WordEmbeddings()
    
    # Train Word2Vec model
    print("Training Word2Vec model...")
    model = embeddings.train_word2vec(processed_sentences, vector_size=100, window=5, min_count=1)
    
    # Find similar words
    test_words = ['language', 'learning', 'neural', 'classification', 'translation']
    for word in test_words:
        print(f"\nWords similar to '{word}':")
        similar_words = embeddings.find_similar_words(word, n=5)
        for similar_word, score in similar_words:
            print(f"  {similar_word}: {score:.4f}")
    
    # Solve analogies
    analogies = [
        ('language', 'processing', 'text'),
        ('machine', 'learning', 'deep'),
        ('neural', 'networks', 'deep')
    ]
    
    print("\nWord Analogies:")
    for word1, word2, word3 in analogies:
        print(f"\n{word1} : {word2} :: {word3} : ?")
        results = embeddings.analogy(word1, word2, word3, n=3)
        for word, score in results:
            print(f"  {word}: {score:.4f}")
    
    # Visualize embeddings
    print("\nVisualizing embeddings...")
    words_to_visualize = ['natural', 'language', 'processing', 'machine', 'learning', 
                          'deep', 'neural', 'networks', 'classification', 'translation']
    
    # Create visualizations with both t-SNE and PCA
    fig1 = embeddings.visualize_embeddings(words_to_visualize, method='tsne')
    fig1.savefig('word_embeddings_tsne.png')
    
    fig2 = embeddings.visualize_embeddings(words_to_visualize, method='pca')
    fig2.savefig('word_embeddings_pca.png')
    
    print("Visualizations saved as 'word_embeddings_tsne.png' and 'word_embeddings_pca.png'")
    
    # Save the model
    model_path = 'word2vec_model'
    embeddings.save_model(model_path)
    print(f"Model saved to '{model_path}'")

if __name__ == "__main__":
    main() 
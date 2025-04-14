# Word Embeddings in NLP

This section covers word embeddings, which are dense vector representations of words in a continuous vector space. Word embeddings capture semantic relationships between words and are fundamental to many NLP tasks.

## Key Concepts

### 1. What are Word Embeddings?

Word embeddings are numerical representations of words in a continuous vector space where:
- Similar words are mapped to nearby points
- Vector operations can capture semantic relationships
- The dimensionality is much lower than one-hot encoding

### 2. Types of Word Embeddings

#### Word2Vec
- Developed by Google
- Two architectures:
  - Continuous Bag of Words (CBOW)
  - Skip-gram
- Predicts context given word or word given context
- Efficient for large datasets

#### GloVe (Global Vectors)
- Developed by Stanford
- Uses global matrix factorization
- Incorporates global context
- Often performs better on analogy tasks

#### FastText
- Extension of Word2Vec by Facebook
- Incorporates subword information
- Better for morphologically rich languages
- Handles out-of-vocabulary words better

### 3. Properties of Word Embeddings

- **Semantic Similarity**: Words with similar meanings have similar vectors
- **Analogies**: Vector operations can solve word analogies (e.g., king - man + woman ≈ queen)
- **Contextual Relationships**: Captures contextual relationships between words
- **Dimensionality**: Typically 50-300 dimensions

## Implementation

The `WordEmbeddings` class in `word_embeddings.py` provides functionality for:

1. Training custom Word2Vec models
2. Loading pre-trained models
3. Converting GloVe embeddings to Word2Vec format
4. Finding similar words
5. Solving word analogies
6. Visualizing embeddings

### Usage Example

```python
from word_embeddings import WordEmbeddings

# Initialize
embeddings = WordEmbeddings()

# Train on your own data
sentences = [['this', 'is', 'a', 'sentence'], ['another', 'sentence', 'here']]
model = embeddings.train_word2vec(sentences)

# Or load a pre-trained model
# embeddings.load_model('path_to_model')

# Find similar words
similar_words = embeddings.find_similar_words('learning', n=5)

# Solve analogies
analogy_result = embeddings.analogy('king', 'queen', 'man')

# Visualize embeddings
words = ['natural', 'language', 'processing', 'machine', 'learning']
fig = embeddings.visualize_embeddings(words)
```

## Visualization

Word embeddings can be visualized using dimensionality reduction techniques:
- t-SNE (t-Distributed Stochastic Neighbor Embedding)
- PCA (Principal Component Analysis)

These techniques reduce high-dimensional vectors to 2D or 3D for visualization while preserving relationships.

## Applications

1. **Text Classification**: Using embeddings as features
2. **Sentiment Analysis**: Capturing sentiment-related word relationships
3. **Machine Translation**: Aligning words across languages
4. **Information Retrieval**: Finding semantically similar documents
5. **Question Answering**: Matching questions to relevant answers

## Best Practices

1. **Pre-trained vs. Custom**: Use pre-trained embeddings for general tasks, train custom ones for domain-specific tasks
2. **Dimensionality**: Choose appropriate dimensionality based on vocabulary size and task
3. **Context Window**: Adjust based on the semantic scope of your task
4. **Training Data**: Ensure training data is representative of your domain
5. **Evaluation**: Use both intrinsic (analogies, similarity) and extrinsic (task performance) evaluation

## Next Steps

After understanding word embeddings, you can move on to:
1. Sequence Models (RNN, LSTM, GRU)
2. Transformer Architecture
3. Advanced NLP Tasks

## Resources

- [Word2Vec Paper](https://arxiv.org/abs/1301.3781)
- [GloVe Paper](https://nlp.stanford.edu/pubs/glove.pdf)
- [FastText Paper](https://arxiv.org/abs/1607.04606)
- [Gensim Documentation](https://radimrehurek.com/gensim/) 
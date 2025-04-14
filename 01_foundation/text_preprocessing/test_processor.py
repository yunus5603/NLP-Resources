from text_processor import TextProcessor

def test_text_processor():
    """
    Test the TextProcessor class with various examples.
    """
    # Initialize the processor
    processor = TextProcessor()
    
    # Test cases
    test_cases = [
        {
            "name": "Basic cleaning",
            "text": "Hello, World! 123",
            "expected": ["hello", "world"]
        },
        {
            "name": "Stopword removal",
            "text": "The quick brown fox jumps over the lazy dog",
            "expected": ["quick", "brown", "fox", "jump", "lazy", "dog"]
        },
        {
            "name": "Lemmatization",
            "text": "running ran runs",
            "expected": ["running", "run", "run"]
        },
        {
            "name": "Complex text",
            "text": """
            Natural Language Processing (NLP) is a subfield of artificial intelligence 
            that focuses on the interaction between computers and human language. 
            It involves the development of algorithms and models that enable computers 
            to understand, interpret, and generate human language in a valuable way.
            """,
            "expected": ["natural", "language", "processing", "subfield", "artificial", 
                        "intelligence", "focus", "interaction", "computer", "human", 
                        "language", "involve", "development", "algorithm", "model", 
                        "enable", "computer", "understand", "interpret", "generate", 
                        "human", "language", "valuable", "way"]
        }
    ]
    
    # Run tests
    for test in test_cases:
        print(f"\nTesting: {test['name']}")
        print("-" * 50)
        print(f"Input text: {test['text']}")
        
        # Process the text
        processed_tokens = processor.process_text(test['text'])
        
        print(f"Processed tokens: {processed_tokens}")
        print(f"Expected tokens: {test['expected']}")
        
        # Verify results
        if processed_tokens == test['expected']:
            print("✓ Test passed!")
        else:
            print("✗ Test failed!")
            print("Differences:")
            print(f"Missing tokens: {set(test['expected']) - set(processed_tokens)}")
            print(f"Extra tokens: {set(processed_tokens) - set(test['expected'])}")

if __name__ == "__main__":
    test_text_processor() 
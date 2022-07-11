import pandas as pd
import re

def generate_ngrams(s, n):
    # Convert to lowercases
    s = s.lower()
    
    # Replace all none alphanumeric characters with spaces
    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
    
    # Break sentence in the token, remove empty tokens
    tokens = [token for token in s.split(" ") if token != ""]
    
    # Use the zip function to help us generate n-grams
    # Concatentate the tokens into ngrams and return
    ngrams = zip(*[tokens[i:] for i in range(n)])
    a = [" ".join(ngram) for ngram in ngrams]
    return a

def n_gram(data):

    # Generate n-grams
    unigram = generate_ngrams(data, 1)
    bigram = generate_ngrams(data, 2)
    trigram = generate_ngrams(data, 3)

    return data
from nltk import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def preprocessing(text):
    #tokenize
    text = text.lower()
    text = word_tokenize(text)
    text = [word for word in text if word.isalpha()]

    #filtering
    list_stopwords = set(stopwords.words('indonesian'))
    tokens_without_stopword = [word for word in text if not word in list_stopwords]

    #stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    list_tokens = tokens_without_stopword
    output   = [stemmer.stem(token) for token in list_tokens]

    return output
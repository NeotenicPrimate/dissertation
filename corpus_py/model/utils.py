import polars as pl
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

stopwords = stopwords.words('english')
stopwords = stopwords + ['the']

def clean_text(series: pl.Series) -> pl.Series:
    tokenizer = nltk.WordPunctTokenizer()
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    texts = []
    for text in series:
        # print(text)
        words = []
        for word in tokenizer.tokenize(text):
            # word = lemmatizer.lemmatize(word)
            word = stemmer.stem(word)
            if len(word) > 2 and word not in stopwords:
                words.append(word)
        texts.append(' '.join(words))
    return pl.Series(texts)
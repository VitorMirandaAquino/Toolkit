import re
import unicodedata
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer, WordNetLemmatizer
from nltk import tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Técnicas de Limpeza de Dados

def lowercase(text):
    """
    Converte o texto para minúsculas.

    Args:
        text (str): O texto de entrada.

    Returns:
        str: Texto em minúsculas.
    """
    return text.lower()

def remove_accents(text):
    """
    Remove acentos e caracteres especiais do texto.

    Args:
        text (str): O texto de entrada.

    Returns:
        str: Texto sem acentos ou caracteres especiais.
    """
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    return text

def remove_stopwords_portuguese(text):
    """
    Remove stopwords do texto.

    Args:
        text (str): O texto de entrada.

    Returns:
        str: Texto sem stopwords.
    """
    stop_words = set(stopwords.words('portuguese')) # Você pode alterar para o seu idioma
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def remove_stopwords_english(text):
    """
    Remove stopwords do texto.

    Args:
        text (str): O texto de entrada.

    Returns:
        str: Texto sem stopwords.
    """
    stop_words = set(stopwords.words('english')) # Você pode alterar para o seu idioma
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def remove_punctuation(text):
    """
    Remove pontuações do texto.

    Args:
        text (str): O texto de entrada.

    Returns:
        str: Texto sem pontuações.
    """
    text = re.sub(r'[^\w\s]', '', text)
    return text

def remove_extra_spaces(text):
    """
    Remove espaços em branco extras, deixando apenas um espaço entre as palavras.

    Args:
        text (str): O texto de entrada.

    Returns:
        str: Texto com espaços em branco extras removidos.
    """
    return re.sub(r'\s+', ' ', text).strip()

def stemming(text):
    """
    Realiza stemming nas palavras do texto.

    Args:
        text (str): O texto de entrada.

    Returns:
        str: Texto com stemming aplicado.
    """
    stemmer = RSLPStemmer()
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

def lemmatization(text):
    """
    Realiza lemmatization nas palavras do texto.

    Args:
        text (str): O texto de entrada.

    Returns:
        str: Texto com lemmatization aplicado.
    """
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

# Formas de Tokenização

def tokenize_whitespace(text):
    """
    Realiza tokenização com base nos espaços em branco usando NLTK.

    Args:
        text (str): O texto de entrada.

    Returns:
        list: Lista de tokens.
    """
    tokenizer = tokenize.WhitespaceTokenizer()
    tokens = tokenizer.tokenize(text)
    return tokens



def tokenize_punctuation(text):
    """
    Realiza tokenização com base na pontuação usando NLTK.

    Args:
        text (str): O texto de entrada.

    Returns:
        list: Lista de tokens.
    """
    tokenizer = tokenize.WordPunctTokenizer()
    tokens = tokenizer.tokenize(text)
    return tokens

# Vetorização

def bag_of_words(corpus):
    """
    Realiza vetorização usando Count Vectorizer (Bag of Words).

    Args:
        corpus (list): Lista de documentos de texto.

    Returns:
        tuple: Matriz de features e lista de palavras.
    """
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer.get_feature_names_out()

def tfidf_vectorize(corpus, ngram_range=(1, 1)):
    """
    Realiza vetorização usando TF-IDF Vectorizer.

    Args:
        corpus (list): Lista de documentos de texto.
        ngram_range (tuple, optional): Faixa de n-grams a serem considerados. Padrão é (1, 1).

    Returns:
        tuple: Uma tupla contendo a matriz de features TF-IDF (documentos x palavras) e uma lista de palavras.
    """
    vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer.get_feature_names_out()


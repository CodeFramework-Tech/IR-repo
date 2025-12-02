# # src/preprocess.py
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords

# # Download necessary NLTK data (if not already done)
# nltk.download('punkt')
# nltk.download('stopwords')

# def preprocess_text(text):
#     """
#     Preprocess the text by tokenizing and removing stopwords.
    
#     Args:
#     - text: The input text to preprocess.
    
#     Returns:
#     - list: A list of cleaned tokens.
#     """
#     tokens = word_tokenize(text.lower())  # Tokenize the text to lowercase
#     stop_words = set(stopwords.words('english'))  # Define English stopwords
#     filtered_tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
#     return filtered_tokens






def preprocess_text(text):
    # Example preprocessing: lowercasing, tokenization, removing stopwords, etc.
    # Adjust this function based on your needs
    import re
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words("english"))
    
    # Lowercase text, remove punctuation, and tokenize
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    tokens = word_tokenize(text)
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    
    return tokens

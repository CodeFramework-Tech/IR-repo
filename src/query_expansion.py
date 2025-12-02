from nltk.corpus import wordnet

def expand_query_with_synonyms(query):
    """
    Expands a query by adding synonyms of each term using WordNet.
    
    Args:
    - query (str): The search query.
    
    Returns:
    - set: A set of expanded query terms.
    """
    expanded_terms = set(query.split())
    for term in query.split():
        synonyms = wordnet.synsets(term)
        for syn in synonyms:
            for lemma in syn.lemmas():
                expanded_terms.add(lemma.name())  # Add synonyms to expanded query
    
    return expanded_terms

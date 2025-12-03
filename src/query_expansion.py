from nltk.corpus import wordnet

def expand_query_with_synonyms(query):
   
    expanded_terms = set(query.split())
    for term in query.split():
        synonyms = wordnet.synsets(term)
        for syn in synonyms:
            for lemma in syn.lemmas():
                expanded_terms.add(lemma.name())  
    return expanded_terms

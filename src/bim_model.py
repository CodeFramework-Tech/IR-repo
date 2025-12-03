import math

def calculate_bim_score(query_terms, document_terms, document_length, average_document_length):
    score = 0
    for term in query_terms:
        term_frequency = document_terms.get(term, 0)
        
        # Using a simple binary relevance model (no term frequency adjustments)
        idf = math.log((document_length - term_frequency + 0.5) / (term_frequency + 0.5) + 1)
        score += idf
    
    return score

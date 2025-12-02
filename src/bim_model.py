import math

def calculate_bim_score(query_terms, document_terms, document_length, average_document_length):
    """
    Calculates the Binary Independence Model (BIM) score for a document based on the query.
    
    Args:
    - query_terms (set): A set of terms from the query.
    - document_terms (dict): A dictionary with terms in the document and their term frequencies.
    - document_length (int): The total number of terms in the document.
    - average_document_length (float): The average document length across the collection.
    
    Returns:
    - float: The calculated BIM score for the document.
    """
    score = 0
    for term in query_terms:
        term_frequency = document_terms.get(term, 0)
        
        # Using a simple binary relevance model (no term frequency adjustments)
        idf = math.log((document_length - term_frequency + 0.5) / (term_frequency + 0.5) + 1)
        score += idf
    
    return score

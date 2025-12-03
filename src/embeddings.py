
import json
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import faiss 


df = pd.read_csv("data/news.csv") 


docs = df["Heading"].astype(str).tolist()


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embeddings(text):
    """
    Given a text, generate BERT embeddings for it.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy() 
    return embeddings


print("Generating BERT embeddings...")
doc_embeddings = np.vstack([get_bert_embeddings(doc) for doc in docs])  

np.save("index/embeddings.npy", doc_embeddings)


with open("index/metadata.json", "w") as f:
    json.dump({"documents": docs}, f)

print("Embedding index built successfully!")


dimension = doc_embeddings.shape[1]  
index = faiss.IndexFlatL2(dimension)  


index.add(doc_embeddings)


faiss.write_index(index, "index/faiss_index.index")

print("FAISS index built and saved successfully!")

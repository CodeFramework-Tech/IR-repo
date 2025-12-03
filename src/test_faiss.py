# test_faiss.py

import faiss
import numpy as np


data = np.random.random((100, 768)).astype('float32')  


index = faiss.IndexFlatL2(768)  
index.add(data) 

print(f"FAISS index has {index.ntotal} vectors.")

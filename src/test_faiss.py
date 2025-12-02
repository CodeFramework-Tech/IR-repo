# test_faiss.py

import faiss
import numpy as np

# Generate some random test data (100 vectors of dimension 768)
data = np.random.random((100, 768)).astype('float32')  # 100 vectors of size 768

# Create a FAISS index
index = faiss.IndexFlatL2(768)  # Using L2 distance
index.add(data)  # Add data to the index

# Check the number of vectors in the index
print(f"FAISS index has {index.ntotal} vectors.")

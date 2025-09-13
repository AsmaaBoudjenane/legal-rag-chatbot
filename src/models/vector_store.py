import faiss
import pickle
import numpy as np

class VectorStore:
    def __init__(self):
        self.index = None
        self.mapping = None
        self.chunks = None
    
    def build_faiss_index(self, embeddings, index_path, mapping_path, mapping_list, chunk_texts):
        """Build FAISS index with cosine similarity and save it with ID mapping."""
        # Create FAISS index
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings.numpy().astype('float32'))
        
        # Save index
        faiss.write_index(self.index, index_path)
        
        # Save mapping
        with open(mapping_path, "wb") as f:
            pickle.dump(mapping_list, f)
        
        # Save chunk texts
        chunk_text_path = mapping_path.replace("mapping", "texts")
        with open(chunk_text_path, "wb") as f:
            pickle.dump(chunk_texts, f)
        
        self.mapping = mapping_list
        self.chunks = chunk_texts
        
        print(f"✅ Saved FAISS index to: {index_path}")
        print(f"✅ Saved mapping to: {mapping_path}")
        print(f"✅ Saved chunk texts to: {chunk_text_path}")
    
    def load_index(self, index_path, mapping_path, chunk_text_path):
        """Load FAISS index and mappings."""
        self.index = faiss.read_index(index_path)
        
        with open(mapping_path, "rb") as f:
            self.mapping = pickle.load(f)
        
        with open(chunk_text_path, "rb") as f:
            self.chunks = pickle.load(f)
        
        print("✅ Loaded FAISS index and mappings")
    
    def search(self, query_embedding, top_k=5):
        """Search for similar documents."""
        if self.index is None:
            raise ValueError("Index not loaded. Call load_index() first.")
        
        D, I = self.index.search(query_embedding.astype('float32'), top_k)
        results = [
            (self.chunks[i], float(D[0][idx]), self.mapping[i]) 
            for idx, i in enumerate(I[0])
        ]
        return results
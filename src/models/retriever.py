from src.models.embeddings import EmbeddingModel
from src.models.vector_store import VectorStore

class Retriever:
    def __init__(self, embedding_model_name, device):
        self.embedding_model = EmbeddingModel(embedding_model_name, device)
        self.vector_store = VectorStore()
    
    def load_index(self, index_path, mapping_path, chunk_text_path):
        """Load the FAISS index and mappings."""
        self.vector_store.load_index(index_path, mapping_path, chunk_text_path)
    
    def retrieve(self, query, top_k=5, method="mean"):
        """Retrieve relevant documents for a query."""
        # Embed the query
        query_embedding = self.embedding_model.embed_query(query, method=method)
        
        # Search in vector store
        results = self.vector_store.search(query_embedding, top_k=top_k)
        
        return results, query_embedding
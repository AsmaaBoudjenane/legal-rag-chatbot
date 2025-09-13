#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.preprocessor import LegalDataPreprocessor
from src.data_processing.chunker import DocumentChunker
from src.models.embeddings import EmbeddingModel
from src.models.vector_store import VectorStore
from src.utils.helpers import load_config, setup_device, create_directories

def main():
    # Load configuration and setup
    config = load_config()
    device = setup_device()
    create_directories()
    
    # Initialize components
    preprocessor = LegalDataPreprocessor()
    
    print("🧹 Cleaning data...")
    # Clean legal data
    preprocessor.clean_legal_data(
        config['data']['legal_data_path'],
        config['data']['cleaned_legal_path']
    )
    
    # Clean QA data
    preprocessor.clean_qa_data(
        config['data']['qa_data_path'],
        config['data']['cleaned_qa_path']
    )
    
    print("📝 Creating embeddings...")
    # Initialize embedding model
    embedding_model = EmbeddingModel(config['model']['embedding_model'], device)
    
    # Initialize chunker
    chunker = DocumentChunker(
        embedding_model.tokenizer,
        chunk_size=config['chunking']['chunk_size'],
        chunk_overlap=config['chunking']['chunk_overlap']
    )
    
    # Chunk data
    df_chunked = chunker.chunk_legal_data(config['data']['cleaned_legal_path'])
    chunks, mapping = chunker.flatten_chunks(df_chunked)
    
    print(f"📊 Created {len(chunks)} chunks from legal data")
    
    # Create embeddings
    embeddings = embedding_model.embed_texts(
        chunks, 
        method=config['retrieval']['embedding_method']
    )
    
    print("💾 Building and saving FAISS index...")
    # Build and save index
    vector_store = VectorStore()
    vector_store.build_faiss_index(
        embeddings=embeddings,
        index_path="data/processed/legal_faiss.index",
        mapping_path="data/processed/legal_chunk_mapping.pkl",
        mapping_list=mapping,
        chunk_texts=chunks
    )
    
    print("✅ Index building completed successfully!")

if __name__ == "__main__":
    main()
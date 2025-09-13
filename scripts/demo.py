#!/usr/bin/env python3
"""
Quick demo script to test the RAG model
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.rag_model import LegalRAGModel
from src.utils.helpers import load_config, setup_device

def main():
    # Load configuration
    config = load_config()
    device = setup_device()
    
    # Initialize RAG model
    rag_config = {
        'embedding_model': config['model']['embedding_model'],
        'generator_model_path': "models/trained/finetuned_aragpt",
        'device': device
    }
    
    print("🤖 Loading RAG model...")
    rag_model = LegalRAGModel(rag_config)
    rag_model.load_models(
        "data/processed/legal_faiss.index",
        "data/processed/legal_chunk_mapping.pkl",
        "data/processed/legal_chunk_texts.pkl"
    )
    
    # Example questions
    questions = [
        "ما هي المواد القانونية المطبقة في قضية حكم قاضي بالتصحيح؟",
        "ما هو الحكم في قضية عقد الإيجار؟",
        
    ]
    
    print("📝 Testing RAG model with example questions:")
    print("="*60)
    
    for i, question in enumerate(questions, 1):
        print(f"\n🧾 Question {i}: {question}")
        answer = rag_model.generate_answer(question)
        print(f"📜 Answer: {answer}")
        print("-"*60)

if __name__ == "__main__":
    main()
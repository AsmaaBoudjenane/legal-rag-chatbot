#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.models.rag_model import LegalRAGModel
from src.evaluation.evaluator import RAGEvaluator
from src.utils.helpers import load_config, setup_device

def main():
    # Load configuration
    config = load_config()
    device = setup_device()
    
    # Load QA data
    qa_df = pd.read_excel(config['data']['cleaned_qa_path'])
    
    # Initialize RAG model
    rag_config = {
        'embedding_model': config['model']['embedding_model'],
        'generator_model_path': "models/trained/finetuned_aragpt",
        'device': device
    }
    
    rag_model = LegalRAGModel(rag_config)
    rag_model.load_models(
        "data/processed/legal_faiss.index",
        "data/processed/legal_chunk_mapping.pkl",
        "data/processed/legal_chunk_texts.pkl"
    )
    
    # Initialize evaluator
    evaluator = RAGEvaluator()
    
    print("🔍 Evaluating Retriever...")
    # Evaluate retriever
    retrieval_metrics = evaluator.evaluate_retriever(
        qa_df, rag_model, k_values=[10, 20, 50, 70]
    )
    
    # Plot retrieval metrics
    evaluator.plot_retrieval_metrics(retrieval_metrics)
    
    print("📝 Evaluating Generator...")
    # Evaluate generator
    generation_metrics = evaluator.evaluate_generator(qa_df, rag_model)
    
    # Save results
    import json
    results = {
        'retrieval_metrics': retrieval_metrics,
        'generation_metrics': {
            'BLEU': generation_metrics['BLEU'],
            'BERTScore_F1': generation_metrics['BERTScore_F1']
        }
    }
    
    with open("results/evaluation_results.json", "w", encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("✅ Evaluation completed! Results saved to results/evaluation_results.json")

if __name__ == "__main__":
    main()

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from bert_score import score
import evaluate

class RAGEvaluator:
    def __init__(self):
        self.bleu = evaluate.load("sacrebleu")
    
    def evaluate_retriever(self, qa_df, rag_model, k_values=[10, 20, 50]):
        """Evaluate retriever performance."""
        all_metrics = {}
        
        for k in k_values:
            print(f"\n🔍 Evaluating @ top-{k}")
            recall, mrr, hits = 0, 0, 0
            total = len(qa_df)
            
            for _, row in tqdm(qa_df.iterrows(), total=total):
                question = row["question"]
                true_case_id = row["case_id"]
                
                # Get retrieval results
                results, _ = rag_model.retriever.retrieve(question, top_k=k)
                retrieved_ids = [result[2] for result in results]  # case_id is at index 2
                
                relevant_found = [cid == true_case_id for cid in retrieved_ids]
                
                if any(relevant_found):
                    recall += 1
                    hits += 1
                    rank = relevant_found.index(True) + 1
                    mrr += 1 / rank
            
            metrics = {
                "Recall@k": recall / total,
                "MRR@k": mrr / total,
                "Hit@k": hits / total
            }
            
            all_metrics[k] = metrics
            for key, value in metrics.items():
                print(f"{key}: {value:.4f}")
        
        return all_metrics
    
    def evaluate_generator(self, qa_df, rag_model):
        """Evaluate generator performance."""
        # Generate answers
        generated_answers = []
        for question in tqdm(qa_df['question'], desc="Generating answers"):
            answer = rag_model.generate_answer(question)
            generated_answers.append(answer)
        
        refs = qa_df['answer'].tolist()
        preds = generated_answers
        
        # BERTScore
        P, R, F1 = score(preds, refs, lang="ar")
        bertscore_f1_avg = F1.mean().item()
        
        # BLEU Score
        bleu_result = self.bleu.compute(
            predictions=preds, 
            references=[[ref] for ref in refs]
        )
        
        results = {
            "BLEU": bleu_result['score'],
            "BERTScore_F1": bertscore_f1_avg,
            "generated_answers": generated_answers
        }
        
        print(f"\n📊 Generator Evaluation Results:")
        print(f"🟦 BLEU Score: {results['BLEU']:.2f}")
        print(f"💡 BERTScore F1: {results['BERTScore_F1']:.4f}")
        
        return results
    
    def plot_retrieval_metrics(self, all_metrics):
        """Plot retrieval evaluation metrics."""
        k_values = list(all_metrics.keys())
        metric_names = ["Recall@k", "MRR@k", "Hit@k"]
        bar_width = 0.25
        x = np.arange(len(k_values))
        
        plt.figure(figsize=(10, 6))
        
        for i, metric in enumerate(metric_names):
            values = [all_metrics[k][metric] for k in k_values]
            plt.bar(x + i * bar_width, values, width=bar_width, label=metric)
        
        plt.title("📊 Arabic Legal Retriever Evaluation")
        plt.xlabel("Top-K")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.xticks(x + bar_width, k_values)
        plt.grid(axis='y')
        plt.legend()
        plt.tight_layout()
        plt.show()
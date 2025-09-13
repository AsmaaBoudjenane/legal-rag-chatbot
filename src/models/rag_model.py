from src.models.retriever import Retriever
from src.models.generator import LegalGenerator

class LegalRAGModel:
    def __init__(self, config):
        self.config = config
        self.device = config.get('device', 'cpu')
        
        # Initialize retriever and generator
        self.retriever = Retriever(
            config['embedding_model'], 
            self.device
        )
        self.generator = LegalGenerator(
            config['generator_model_path'], 
            self.device
        )
    
    def load_models(self, index_path, mapping_path, chunk_text_path):
        """Load the retrieval index."""
        self.retriever.load_index(index_path, mapping_path, chunk_text_path)
    
    def generate_answer(self, question, top_k=3, threshold=0.6):
        """Complete RAG pipeline for question answering."""
        # Check if question is valid
        if not self.generator.is_legal_arabic_question(question):
            return "❌ عذراً، لا يمكنني الإجابة على هذا السؤال لأنه خارج النطاق القانوني أو ليس مكتوباً بالللغة العربية القانونية."
        
        # Step 1: Retrieve relevant documents
        retrieval_results, _ = self.retriever.retrieve(
            query=question, top_k=top_k
        )
        
        # Check retrieval quality
        if not retrieval_results or retrieval_results[0][1] < threshold:
            return "❌ عذراً، لا أمتلك معلومات كافية للإجابة عن هذا السؤال."
        
        # Step 2: Prepare context
        context = "\n".join(res[0] for res in retrieval_results)
        
        # Step 3: Create prompt based on question type
        if self.generator.is_law_article_question(question):
            prompt = (
                f"المعرفة التالية مستخلصة من قضايا قانونية:\n{context}\n\n"
                f"استناداً إليها، ما هي **جميع المواد القانونية** التي يجب استخدامها لحل القضية التالية:\n"
                f"{question}\n\n"
                f"استخرج المواد القانونية الحقيقية التي وردت في النص، ولا تكرر أمثلة وهمية.\n"
                f"الجواب:"
            )
            max_tokens = 100
        else:
            prompt = f"{question}\n الجواب:\n{context}"
            max_tokens = 250
        
        # Step 4: Generate answer
        generated_output = self.generator.generate(prompt, max_new_tokens=max_tokens)
        
        # Step 5: Post-process answer
        if self.generator.is_law_article_question(question):
            return self.generator.extract_articles_with_law(generated_output)
        else:
            final_answer = self.generator.smart_clean_generated_answer(
                generated_output, question
            )
            return final_answer if final_answer.strip() else "❌ عذراً، لا أمتلك معلومات كافية للإجابة عن هذا السؤال."

import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

class LegalGenerator:
    def __init__(self, model_path, device):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.device = device
        
        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def is_legal_arabic_question(self, text):
        """Check if text is a legal Arabic question."""
        if not text.strip():
            return False
        
        # Calculate Arabic character percentage
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        arabic_ratio = arabic_chars / len(text) if len(text) > 0 else 0
        
        legal_keywords = [
            'قررت', 'القضية', 'قانون', 'محكمة', 'عقوبة', 'عقد', 
            'دعوى', 'حكم', 'مدني', 'المواد'
        ]
        contains_legal_terms = any(kw in text for kw in legal_keywords)
        
        return arabic_ratio > 0.5 and contains_legal_terms
    
    def is_law_article_question(self, question):
        """Check if question is asking about law articles."""
        keywords = [
            'ما هي المواد', 'ما المواد', 'المواد القانونية', 'أي مواد',
            'ما المادة', 'ما هي المادة', 'النص القانوني'
        ]
        return any(kw in question for kw in keywords)
    
    def extract_articles_with_law(self, text):
        """Extract legal articles from generated text."""
        pattern = r"(?:المادة|المواد)\s+([\d\sوو\-إلى]+)\s*(?:من\s+(قانون\s+[^\nو\.]*))?"
        matches = re.findall(pattern, text)
        
        output = []
        for articles, law in matches:
            law_name = law.strip() if law else ""
            cleaned_articles = articles.replace("-", "إلى").replace("—", "إلى").replace("  ", " ").strip()
            output.append(f"المواد {cleaned_articles} {law_name}")
        
        return "✅ المواد القانونية المستخرجة:\n" + "\n".join(output)
    
    def smart_clean_generated_answer(self, answer, question):
        """Clean the generated answer."""
        # Remove the full question if it appears inside the answer
        answer = answer.replace(question, "").strip()
        
        # Remove the word "الجواب" if it appears
        answer = answer.replace("الجواب", "").strip()
        
        # Remove leading ":" if any
        if answer.startswith(":"):
            answer = answer[1:].strip()
        
        return answer
    
    def generate(self, prompt, max_new_tokens=250):
        """Generate text based on prompt."""
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, padding=True
        ).to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=4,
                no_repeat_ngram_size=2,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        output = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return output
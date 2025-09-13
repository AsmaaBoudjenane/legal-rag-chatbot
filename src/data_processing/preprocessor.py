import pandas as pd
from src.utils.helpers import normalize_arabic

class LegalDataPreprocessor:
    def __init__(self):
        pass
    
    def clean_legal_data(self, input_path, output_path):
        """Normalize columns of the legal data Excel and save cleaned version."""
        df = pd.read_excel(input_path)
        columns_to_clean = ['Case ID', 'Title', 'Keywords', 'Description']
        
        for col in columns_to_clean:
            if col in df.columns:
                df[col] = df[col].apply(normalize_arabic)
        
        df.to_excel(output_path, index=False)
        print(f"✅ Cleaned legal data saved to: {output_path}")
        return df
    
    def clean_qa_data(self, input_path, output_path):
        """Normalize question, answer, and context in the QA dataset."""
        df = pd.read_excel(input_path)
        columns_to_clean = ['question', 'answer', 'context']
        
        for col in columns_to_clean:
            if col in df.columns:
                df[col] = df[col].apply(normalize_arabic)
        
        df.to_excel(output_path, index=False)
        print(f"✅ Cleaned QA data saved to: {output_path}")
        return df
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.utils.helpers import normalize_arabic

class DocumentChunker:
    def __init__(self, tokenizer, chunk_size=300, chunk_overlap=50):
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_legal_data(self, input_path):
        """Load legal data, apply normalization, and split into overlapping chunks."""
        df = pd.read_excel(input_path)
        
        # Normalize data
        for col in ['Case ID', 'Title', 'Keywords', 'Description']:
            if col in df.columns:
                df[col] = df[col].apply(normalize_arabic)
        
        def chunk_row(row):
            combined = " - ".join([
                str(row[c]) for c in ['Title', 'Keywords', 'Description']
                if c in row and pd.notnull(row[c])
            ])
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ".", "؟", ":", "؛", "،", "-"],
                length_function=lambda text: len(self.tokenizer.tokenize(text))
            )
            return splitter.split_text(combined)
        
        df['case_chunks'] = df.apply(chunk_row, axis=1)
        return df
    
    def flatten_chunks(self, df):
        """Convert chunked DataFrame into list of texts and list of corresponding case IDs."""
        all_chunks, id_map = [], []
        
        for _, row in df.iterrows():
            if isinstance(row['case_chunks'], list):
                for chunk in row['case_chunks']:
                    all_chunks.append(chunk)
                    id_map.append(row['Case ID'])
        
        return all_chunks, id_map
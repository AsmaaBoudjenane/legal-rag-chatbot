import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.rag_model import LegalRAGModel
from src.utils.helpers import load_config

class LegalRAGInterface:
    def __init__(self):
        self.config = load_config()
        self.rag_model = None
        
    @st.cache_resource
    def load_model(_self):
        """Load the RAG model (cached)."""
        config = {
            'embedding_model': _self.config['model']['embedding_model'],
            'generator_model_path': "finetuned_aragpt",
            'device': _self.config['model']['device']
        }
        
        model = LegalRAGModel(config)
        model.load_models(
            "legal_faiss.index",
            "legal_chunk_mapping.pkl", 
            "legal_chunk_texts.pkl"
        )
        return model
    
    def run(self):
        """Run the Streamlit interface."""
        st.set_page_config(
            page_title="المساعد القانوني الذكي", 
            page_icon="⚖️",
            layout="centered"
        )
        
        # Custom CSS for Arabic text
        st.markdown("""
        <style>
        .stTextInput > div > div > input {
            direction: rtl;
            text-align: right;
        }
        .stTextArea > div > div > textarea {
            direction: rtl;
            text-align: right;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Title
        st.title("🧠 المساعد القانوني الذكي")
        st.markdown("نظام ذكي للإجابة عن الأسئلة القانونية بالللغة العربية")
        
        # Load model
        if self.rag_model is None:
            with st.spinner("جاري تحميل النموذج..."):
                self.rag_model = self.load_model()
        
        # Input section
        question = st.text_area(
            "🧾 أدخل سؤالك القانوني:",
            placeholder="مثال: ما هي المواد القانونية في قضية عقد الإيجار؟",
            height=100
        )
        
        # Generate answer
        if st.button("🔍 تحليل السؤال القانوني", type="primary"):
            if question.strip():
                with st.spinner("جاري البحث والتحليل..."):
                    answer = self.rag_model.generate_answer(question)
                
                st.markdown("### 📜 الجواب القانوني:")
                st.text_area(
                    "", 
                    value=answer,
                    height=200,
                    disabled=True
                )
            else:
                st.warning("يرجى إدخال سؤال قانوني")
        
        # Clear button
        if st.button("🧹 مسح"):
            st.rerun()

if __name__ == "__main__":
    app = LegalRAGInterface()
    app.run()

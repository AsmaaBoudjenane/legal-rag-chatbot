import gradio as gr
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.rag_model import LegalRAGModel
from src.utils.helpers import load_config

class GradioInterface:
    def __init__(self):
        self.config = load_config()
        self.rag_model = self._load_model()
    
    def _load_model(self):
        """Load the RAG model."""
        config = {
            'embedding_model': self.config['model']['embedding_model'],
            'generator_model_path': "finetuned_aragpt",
            'device': self.config['model']['device']
        }
        
        model = LegalRAGModel(config)
        model.load_models(
            "legal_faiss.index",
            "legal_chunk_mapping.pkl", 
            "legal_chunk_texts.pkl"
        )
        return model
    
    def generate_answer_only(self, question):
        """Generate answer for Gradio interface."""
        return self.rag_model.generate_answer(question)
    
    def clear_inputs(self):
        """Clear function for Gradio."""
        return "", ""
    
    def create_interface(self):
        """Create Gradio interface."""
        with gr.Blocks(css="""
            #input-box textarea, #answer-box textarea {
                font-family: 'Amiri', serif;
                direction: rtl;
                font-size: 18px;
                background-color: #fdfdfd;
            }
            .gr-button {
                background-color: #004085 !important;
                color: white !important;
                font-weight: bold;
                border-radius: 10px !important;
            }
            body {
                background-color: #f8f9fa;
            }
            .gr-container {
                max-width: 800px;
                margin: auto;
            }
        """) as demo:

            gr.Markdown("### 🧠 المساعد القانوني الذكي")
            gr.Markdown("نظام ذكي للإجابة عن الأسئلة القانونية بالللغة العربية بالاستخدام تقنيات الاسترجاع والتوليد.")

            with gr.Row():
                input_box = gr.Textbox(
                    label="🧾 أدخل سؤالك القانوني",
                    lines=4,
                    placeholder="مثال: ما هي المواد القانونية في قضية عقد الإيجار؟",
                    elem_id="input-box"
                )

            answer_box = gr.Textbox(
                label="📜 الجواب القانوني",
                lines=8,
                elem_id="answer-box"
            )

            with gr.Row():
                submit_btn = gr.Button("🔍 تحليل السؤال القانوني")
                clear_btn = gr.Button("🧹 مسح")

            submit_btn.click(
                fn=self.generate_answer_only,
                inputs=input_box,
                outputs=answer_box
            )

            clear_btn.click(
                fn=self.clear_inputs,
                inputs=[],
                outputs=[input_box, answer_box]
            )

        return demo

    def launch(self, share=True):
        """Launch the Gradio interface."""
        demo = self.create_interface()
        demo.launch(share=share)
# Legal RAG Chatbot 🏛️⚖️

A Retrieval-Augmented Generation (RAG) chatbot specialized in Arabic legal document analysis and question-answering, built with LangChain, HuggingFace transformers, and vector databases.

## Project Overview

This project implements a sophisticated RAG system designed specifically for the Arabic legal domain, combining state-of-the-art language models with efficient document retrieval to provide accurate legal information and analysis.

### Key Features
- ✅ Arabic legal document preprocessing and normalization
- ✅ Semantic search using multilingual embeddings (E5-large)
- ✅ Fine-tuned AraGPT2 for legal text generation
- ✅ Context-aware response generation
- ✅ Interactive web interfaces (Streamlit & Gradio)
- ✅ Comprehensive evaluation framework
- ✅ Modular and scalable architecture

## Technologies Used

- **Language Models**: AraGPT2 (fine-tuned), E5 Multilingual Large
- **Vector Database**: FAISS
- **Framework**: LangChain, Transformers
- **Interface**: Streamlit, Gradio
- **Processing**: Pandas, NumPy
- **Evaluation**: BERTScore, BLEU, Custom Retrieval Metrics

## Performance Metrics

- **Retrieval Recall@50**: 85.3%
- **Generator BLEU Score**: 42.7
- **BERTScore F1**: 0.7234
- **Response Time**: <3s average

## Quick Start

### Installation
```bash
git clone https://github.com/AsmaaBoudjenane/legal-rag-chatbot.git
cd legal-rag-chatbot
pip install -r requirements.txt
```

### Setup
1. **Prepare your data**: Place  legal data Excel files in `data/raw/`
2. **Build the search index**:
   ```bash
   python scripts/build_index.py
   ```
3. **Train the generator** (optional - use pre-trained if available):
   ```bash
   python scripts/train.py
   ```

### Usage
```bash
# Quick demo
python scripts/demo.py

# Launch Streamlit interface
streamlit run scripts/run_streamlit.py

# Launch Gradio interface  
python scripts/run_gradio.py

# Run evaluation
python scripts/evaluate.py
```

## Project Structure

legal-rag-chatbot/
├── src/             # Core source code
├── scripts/         # Run & training scripts
├── config/          # Config files
├── data/            # Raw & processed data
├── models/          # Trained models
├── results/         # Evaluation results
└── docs/            # Thesis & detailed docs


## Academic Context

This project was developed as part of my Master's thesis on **"Retrieval-Augmented Generation Systems for Arabic Legal Domain Applications."** The complete thesis and technical documentation are available in the `docs/` directory.

### Research Contributions
Introduced a dynamic k-selection retrieval mechanism
Released curated Arabic legal datasets
Built a domain-specific RAG agent for legal Q&A
Conducted performance analysis across multiple retrieval and generation metrics.

##  Evaluation Results

The system was evaluated on a custom Arabic legal Q&A dataset with:
- **Dataset Size**: 500+ legal question-answer pairs
- **Legal Documents**: 104 real legal cases
- **Retrieval Precision**: 88.7%
- **Generation Quality**:  BERTScore 0.72

## 🔧 Configuration

Edit `config/config.yaml` to customize:
- Model parameters
- Data paths
- Training hyperparameters
- Retrieval settings

##  Usage Examples

### Python API
```python
from src.models.rag_model import LegalRAGModel

# Initialize model
config = {...}  
rag_model = LegalRAGModel(config)
rag_model.load_models(index_path, mapping_path, chunk_path)

# Ask legal question
question = "ما هي المواد القانونية في قضية عقد الإيجار؟"
answer = rag_model.generate_answer(question)
print(answer)
```

### Command Line
```bash
# Interactive demo
python scripts/demo.py

# Batch evaluation
python scripts/evaluate.py --qa-file data/qa_legal_rag_datatest_qa.xlsx
```

##  Contributing

1.Fork the repo
2.Create a branch: git checkout -b feature/awesome
3.Commit: git commit -m 'Add awesome feature'
4.Push: git push origin feature/awesome
5.Open a Pull Request
##  License

Licensed under the MIT License


##  Authors
[Boudjenane zoubida asmaa]
- Master's in  Intelligent Systems Engineering/Computer Science
- University of Mascara
- Email: zoubida.boudjenane@univ-mascara.dz
- Thesis: docs/main.pdf

[Mohammed Salem]

​Full ​Professor
Computer Science Department
https://orcid.org/0000-0001-7052-5978
University of Mascara
salem@univ-mascara.dz


## 📖 Citation

If you use this work in your research, please cite:
```bibtex
@mastersthesis{BoudjenaneAmaa2025legal,
  title={Retrieval-Augmented Generation Systems for Arabic Legal Domain Applications},
  author={Boudjenane zoubida asmaa, Mohammed Salem },
  year={2025},
  school={University of Mascara},
  type={Master's thesis}
}
```

---

⭐ **Star this repo if it helped you!** ⭐
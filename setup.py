from setuptools import setup, find_packages

setup(
    name="legal-rag-chatbot",            
    version="0.1.0",                      
    authors= "Boudjenane zoubida asmaa, Mohammed Salem"
    authors_email="zoubida.boudjenane@univ-mascara.dz","salem@univ-mascara.dz"
    description="A Retrieval-Augmented Generation chatbot for Arabic legal documents.",
    long_description=open("README.md", encoding="utf-8").read(),
    url="https://github.com/AsmaaBoudjenane/legal-rag-chatbot.git", 
    packages=find_packages(where="src"),  # look in /src for modules
    package_dir={"": "src"},              # root points to /src
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[                   
        "torch",
        "transformers",
        "langchain",
        "faiss-cpu",       # or faiss-gpu if you need GPU
        "pandas",
        "numpy",
        "streamlit",
        "gradio",
        "bert-score",
        "sacrebleu"
    ],
)

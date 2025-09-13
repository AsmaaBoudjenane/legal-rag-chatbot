import re
import pandas as pd
import torch
import yaml
import os

def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def normalize_arabic(text):
    """
    Normalize Arabic text:
    - Remove diacritics
    - Normalize Alef forms  
    - Convert hamza-related letters to base letters
    - Normalize quotes and whitespace
    """
    if not isinstance(text, str):
        return text
    text = re.sub(r'[ُِّٰۥۦۧۨ۩ۭ]', '', text)  # Remove diacritics (Tashkeel)
    text = re.sub(r'[إأآا]', 'ا', text)  # Normalize Alef
    text = re.sub(r'ؤ', 'و', text)  # Convert hamza-Waw to Waw
    text = re.sub(r'["""]', '"', text)  # Normalize double quotes
    text = re.sub(r'[''']', "'", text)  # Normalize apostrophes
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def setup_device():
    """Setup and return the appropriate device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def create_directories():
    """Create necessary directories for the project."""
    dirs = [
        "data/raw", "data/processed", "data/qa_evaluation",
        "models/trained", "results", "logs"
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
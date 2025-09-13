#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.trainer import LegalModelTrainer
from src.utils.helpers import load_config, create_directories

def main():
    # Load configuration
    config = load_config()
    
    # Create directories
    create_directories()
    
    # Initialize trainer
    trainer = LegalModelTrainer(
        model_name=config['model']['generator_model'],
        output_dir="models/trained/training_results"
    )
    
    # Prepare dataset
    print("📚 Preparing training dataset...")
    dataset = trainer.prepare_dataset(config['data']['cleaned_qa_path'])
    tokenized_dataset = trainer.tokenize_dataset(dataset, max_length=config['training']['max_length'])
    
    # Train model
    print("🚀 Starting training...")
    trained_model = trainer.train(
        dataset=tokenized_dataset,
        batch_size=config['training']['batch_size'],
        epochs=config['training']['epochs']
    )
    
    # Save model
    save_path = "models/trained/finetuned_aragpt"
    trainer.save_model(save_path)
    
    print("✅ Training completed successfully!")

if __name__ == "__main__":
    main()
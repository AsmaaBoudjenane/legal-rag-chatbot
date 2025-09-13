from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, 
    DataCollatorForLanguageModeling
)
import pandas as pd

class LegalModelTrainer:
    def __init__(self, model_name, output_dir):
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_dataset(self, qa_file_path):
        """Prepare dataset from QA Excel file."""
        df = pd.read_excel(qa_file_path)
        
        def format_example(example):
            prompt = f"السؤال: {example['question']}\nالنص المرجعي: {example['context']}\nالجواب:"
            full_text = f"{prompt} {example['answer']}"
            return {"text": full_text}
        
        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(format_example).remove_columns(dataset.column_names)
        
        return dataset
    
    def tokenize_dataset(self, dataset, max_length=1024):
        """Tokenize the dataset."""
        def tokenize_fn(examples):
            return self.tokenizer(
                examples["text"], 
                truncation=True, 
                padding="max_length", 
                max_length=max_length
            )
        
        return dataset.map(tokenize_fn, batched=True)
    
    def train(self, dataset, batch_size=2, epochs=20, save_strategy="epoch"):
        """Train the model."""
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            save_strategy=save_strategy,
            logging_dir="./logs",
            logging_strategy="epoch",
            report_to="none"
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        trainer.train()
        return trainer
    
    def save_model(self, save_path):
        """Save the trained model and tokenizer."""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"✅ Model saved to {save_path}")
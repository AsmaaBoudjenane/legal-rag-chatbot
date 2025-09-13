import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

class EmbeddingModel:
    def __init__(self, model_name, device):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device).eval()
        self.device = device
        
    def mean_pooling(self, model_output, attention_mask):
        """Compute mean pooling over the token embeddings."""
        token_embeddings = model_output.last_hidden_state
        input_mask_exp = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return (token_embeddings * input_mask_exp).sum(1) / input_mask_exp.sum(1)
    
    def embed_texts(self, texts, method="mean", batch_size=8):
        """Embed list of texts using the model and return normalized embeddings."""
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts[i:i+batch_size]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True, 
                return_tensors="pt", max_length=512
            ).to(self.model.device)
            
            with torch.no_grad():
                output = self.model(**encoded)
                if method == "mean":
                    batch_embeds = self.mean_pooling(output, encoded["attention_mask"])
                else:
                    batch_embeds = output.last_hidden_state[:, 0, :]
                batch_embeds = torch.nn.functional.normalize(batch_embeds, p=2, dim=1)
            
            embeddings.append(batch_embeds.cpu())
        
        return torch.cat(embeddings, dim=0)
    
    def embed_query(self, query, method="mean"):
        """Embed a single query for retrieval."""
        query = f"query: {query}"  # E5 format
        inputs = self.tokenizer(
            query, return_tensors="pt", truncation=True, 
            padding=True, max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            output = self.model(**inputs)
            if method == "mean":
                embeddings = self.mean_pooling(output, inputs["attention_mask"])
            else:
                embeddings = output.last_hidden_state[:, 0, :]
            return torch.nn.functional.normalize(embeddings, p=2, dim=1).cpu().numpy()
import torch
import pickle
import re


class NoamScheduler:    
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
    
    def step(self):
        self.step_num += 1
        lr = self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5)
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def state_dict(self):
        return {"step_num": self.step_num}
    
    def load_state_dict(self, state):
        self.step_num = state["step_num"]


class EarlyStopping:
    def __init__(self, patience=3, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def generate_translation(model, eng_tokenizer, rus_tokenizer, input_sentence, max_len=96, device="cuda"):
    """Генерирует перевод для одного предложения."""
    model.eval()
    if device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    with torch.no_grad():
        source_indices = eng_tokenizer.encode(
            input_sentence,
            max_length=max_len,
            add_special_tokens=True
        )
        source_tensor = torch.tensor([source_indices], dtype=torch.long, device=device)
        
        target_indices = [rus_tokenizer.sos_id]
        max_generation_length = min(max_len, max_len - 1)
        
        for _ in range(max_generation_length):
            target_tensor = torch.tensor([target_indices], dtype=torch.long, device=device)
            outputs = model(source_tensor, target_tensor)
            
            next_token_logits = outputs[0, -1, :]
            next_token_logits[rus_tokenizer.pad_id] = -float("inf")
            next_token_logits[rus_tokenizer.sos_id] = -float("inf")
            
            next_token_id = torch.argmax(next_token_logits).item()
            target_indices.append(next_token_id)
            
            if next_token_id == rus_tokenizer.eos_id:
                break
        
        translation = rus_tokenizer.decode(target_indices)
        return translation


def load_model(checkpoint_path, eng_tokenizer_path, rus_tokenizer_path, device="cuda"):
    """Загружает модель и токенизаторы из сохранённых файлов."""
    from .model import TransformerModel
    
    if device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    with open(eng_tokenizer_path, "rb") as f:
        eng_data = pickle.load(f)
        eng_tokenizer = SubWordTokenizer(eng_data["vocabulary"], eng_data["merges"])
    
    with open(rus_tokenizer_path, "rb") as f:
        rus_data = pickle.load(f)
        rus_tokenizer = SubWordTokenizer(rus_data["vocabulary"], rus_data["merges"])
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    
    model = TransformerModel(
        src_vocab_size=config["src_vocab_size"],
        tgt_vocab_size=config["tgt_vocab_size"],
        sequence_length=config["sequence_length"],
        hidden_dim=config["hidden_dim"],
        intermediate_dim=config["intermediate_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    return model, eng_tokenizer, rus_tokenizer

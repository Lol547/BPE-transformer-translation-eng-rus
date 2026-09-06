import torch
from torch.utils.data import Dataset, DataLoader


class TranslationDataset(Dataset):
    def __init__(self, pairs, eng_tokenizer, rus_tokenizer, sequence_length):
        self.pairs = pairs
        self.eng_tokenizer = eng_tokenizer
        self.rus_tokenizer = rus_tokenizer
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        eng, rus = self.pairs[idx]
        
        eng_indices = self.eng_tokenizer.encode(eng, max_length=self.sequence_length, add_special_tokens=True)
        rus_indices = self.rus_tokenizer.encode(rus, max_length=self.sequence_length + 1, add_special_tokens=True)
        
        eng_tensor = torch.tensor(eng_indices, dtype=torch.long)
        rus_tensor = torch.tensor(rus_indices, dtype=torch.long)
        
        features = {
            "english": eng_tensor,
            "russian": rus_tensor[:-1]
        }
        labels = rus_tensor[1:]
        
        return features, labels


def make_dataset(pairs, eng_tokenizer, rus_tokenizer, sequence_length, batch_size=64, shuffle=True):
    """Создаёт DataLoader для TranslationDataset."""
    dataset = TranslationDataset(pairs, eng_tokenizer, rus_tokenizer, sequence_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

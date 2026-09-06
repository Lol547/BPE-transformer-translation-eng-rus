import re
from collections import Counter

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def split_text(text):
    """Разбивает текст на токены (слова и пунктуацию)."""
    return TOKEN_PATTERN.findall(text)


def count_and_split_words(data):
    """Подсчитывает частоту символов в словах."""
    counts = Counter()
    for line in data:
        for word in split_text(line):
            symbols = tuple(list(word) + ["</w>"])
            counts[symbols] += 1
    return counts


def count_pairs(counts):
    """Подсчитывает частоту пар соседних символов."""
    pairs = Counter()
    for symbols, freq in counts.items():
        for pair in zip(symbols[:-1], symbols[1:]):
            pairs[pair] += freq
    return pairs


def merge_pair(counts, pair):
    """Объединяет пару символов в один токен."""
    first, second = pair
    merged = first + second
    new_counts = Counter()
    
    for symbols, freq in counts.items():
        new_symbols = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == first and symbols[i + 1] == second:
                new_symbols.append(merged)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_counts[tuple(new_symbols)] += freq
    
    return new_counts


def compute_sub_word_vocabulary(dataset, vocab_size):
    """
    Вычисляет BPE-словарь на основе датасета.
    
    Возвращает:
        vocab_dict: {token: index}
        merges_dict: {(token1, token2): rank}
    """
    counts = count_and_split_words(dataset)
    char_counts = Counter()
    for symbols, freq in counts.items():
        for symbol in symbols:
            char_counts[symbol] += freq
    
    vocab = ["[PAD]", "[UNK]", "[SOS]", "[EOS]"]
    for symbol, _ in char_counts.most_common():
        if len(vocab) >= vocab_size:
            break
        if symbol not in vocab:
            vocab.append(symbol)
    
    merges = []
    while len(vocab) < vocab_size:
        pairs = count_pairs(counts)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        counts = merge_pair(counts, best_pair)
        merged_token = best_pair[0] + best_pair[1]
        if merged_token not in vocab:
            vocab.append(merged_token)
        merges.append(best_pair)
    
    vocab_dict = {token: index for index, token in enumerate(vocab)}
    merges_dict = {pair: rank for rank, pair in enumerate(merges)}
    
    return vocab_dict, merges_dict


class SubWordTokenizer:
    """BPE-токенизатор с поддержкой кэширования."""
    
    def __init__(self, vocabulary, merges):
        self.vocabulary = vocabulary
        self.token_to_id = vocabulary
        self.merges = merges
        
        self.id_to_token = {idx: token for token, idx in vocabulary.items()}
        
        self.pad_id = vocabulary["[PAD]"]
        self.unk_id = vocabulary["[UNK]"]
        self.sos_id = vocabulary["[SOS]"]
        self.eos_id = vocabulary["[EOS]"]
        
        self.cache = {}
    
    def bpe_merge(self, subwords):
        """Применяет BPE-слияния к последовательности субслов."""
        cache_key = tuple(subwords)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return list(cached)
        
        subwords = list(subwords)
        while len(subwords) > 1:
            best_pair = None
            best_rank = float("inf")
            
            for pair in zip(subwords[:-1], subwords[1:]):
                rank = self.merges.get(pair)
                if rank is not None and rank < best_rank:
                    best_pair = pair
                    best_rank = rank
            
            if best_pair is None:
                break
            
            first, second = best_pair
            merged_subwords = []
            i = 0
            while i < len(subwords):
                if i < len(subwords) - 1 and subwords[i] == first and subwords[i + 1] == second:
                    merged_subwords.append(first + second)
                    i += 2
                else:
                    merged_subwords.append(subwords[i])
                    i += 1
            
            if merged_subwords == subwords:
                break
            subwords = merged_subwords
        
        self.cache[cache_key] = tuple(subwords)
        return subwords
    
    def tokenize(self, text):
        """Разбивает текст на субсловные токены."""
        tokens = []
        for word in split_text(text):
            subwords = list(word) + ["</w>"]
            merged_subwords = self.bpe_merge(subwords)
            tokens.extend(merged_subwords)
        return tokens
    
    def encode(self, text, max_length=None, add_special_tokens=True):
        """Кодирует текст в последовательность индексов."""
        tokens = self.tokenize(text)
        indices = [self.token_to_id.get(token, self.unk_id) for token in tokens]
        
        if add_special_tokens:
            indices = [self.sos_id] + indices + [self.eos_id]
        
        if max_length is not None:
            if len(indices) > max_length:
                indices = indices[:max_length]
                if add_special_tokens:
                    indices[-1] = self.eos_id
            elif len(indices) < max_length:
                indices += [self.pad_id] * (max_length - len(indices))
        
        return indices
    
    def decode(self, indices):
        """Декодирует последовательность индексов в текст."""
        tokens = []
        for idx in indices:
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            token = self.id_to_token.get(idx)
            if token is None or token in {"[PAD]", "[UNK]", "[SOS]"}:
                continue
            if token == "[EOS]":
                break
            tokens.append(token)
        
        words = []
        current_word = ""
        for token in tokens:
            if "</w>" in token:
                current_word += token.replace("</w>", "")
                words.append(current_word)
                current_word = ""
            else:
                current_word += token
        
        if current_word:
            words.append(current_word)
        
        text = " ".join(words)
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        text = re.sub(r"([([])\s+", r"\1", text)
        text = re.sub(r"\s+([)\]])", r"\1", text)
        return text
    
    def __call__(self, text, max_length=None):
        return self.encode(text, max_length=max_length)

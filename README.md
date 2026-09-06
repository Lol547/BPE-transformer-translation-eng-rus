# Transformer Translator - Нейронный перевод с английского на русский

> **Реализация Transformer с нуля на PyTorch для перевода с английского на русский(своя реализация BPE-токенизации и позиционных эмбеддингов). Архитектура: Positional Embedding, Multi-Head Self-Attention, Feed-Forward, Noam-планировщик. Обучена на датасете ManyThings (536k пар). Достигает BLEU 0.45 на тестовой выборке.**

---

## Навигация
- [Особенности](#особенности)
- [Архитектура модели](#архитектура-модели)
- [Результаты](#результаты)
- [Установка](#установка)
- [Использование](#использование)
  - [Инференс](#инференс)
  - [Обучение](#обучение)
- [Структура проекта](#структура-проекта)
- [Веса модели](#веса-модели)

---

## Особенности

- **Transformer с нуля** - полная реализация на PyTorch без использования готовых библиотечных моделей.
- **BPE-токенизация** - кастомная реализация Byte-Pair Encoding (BPE) для эффективной работы с редкими словами.
- **Positional Embedding** - синусоидальное позиционное кодирование, добавляющее информацию о порядке слов.
- **Weight Tying** - классификатор использует те же веса, что и target embedding.
- **Multi-Head Self-Attention** - 8 голов внимания, позволяющих модели фокусироваться на разных аспектах контекста.
- **Noam-планировщик** - динамическое изменение learning rate по схеме из статьи "Attention Is All You Need".
- **Label Smoothing** - сглаживание меток (0.1) для улучшения обобщения.
- **GradScaler + autocast** - смешанная точность для ускорения обучения на GPU.

---

## Архитектура модели

Модель представляет собой классический Transformer (как в статье "Attention Is All You Need"):

| Компонент | Описание |
|-----------|----------|
| **Positional Embedding** | Синусоидальное кодирование позиций (sequence_length=96, hidden_dim=384) |
| **Encoder** | 5 слоёв, каждый с Multi-Head Self-Attention (8 heads) и Feed-Forward (intermediate_dim=1024) |
| **Decoder** | 5 слоёв, каждый с Masked Self-Attention, Cross-Attention и Feed-Forward |
| **Функция потерь** | CrossEntropyLoss с Label Smoothing (0.1) и ignore_index=PAD_IDX |
| **Оптимизатор** | Adam (betas=(0.9, 0.98), eps=1e-9) |
| **Планировщик** | NoamScheduler (warmup_steps=10000) |
| **Регуляризация** | Dropout 0.2, Weight Tying (shared embedding weights) |

**Количество параметров:** ~39.4 млн

---

## Результаты

Модель обучалась 20 эпох (прервана вручную, но лучший чекпоинт на 20-й эпохе). Метрики:

| Метрика | Значение |
|---------|----------|
| **Train Loss** | 2.8099 |
| **Validation Loss** | **2.7546** |
| **Validation Token Accuracy** | **76.37%** |
| **Corpus BLEU** | **0.4545 (45.45%)** |

### Примеры перевода

| Исходный текст (EN) | Перевод (RU) |
|---------------------|--------------|
| Hello, how are you? | Привет, как ты? |
| Although it was late, he continued working. | Несмотря на то, что опоздал, он продолжал работать. |
| What is your name? | Как тебя зовут? |
| The book that you gave me is very interesting. | Книга, которую ты мне дал, очень интересная. |
| She will come tomorrow. | Она придёт завтра. |
| I love programming. | Я люблю путешествовать. |
| Machine learning is interesting. | Сророн - это интересно. |

> **Примечание:** Качество перевода хорошее для коротких предложений, но модель всё ещё допускает ошибки на сложных конструкциях и незнакомых словах(очевидно). Для улучшения я бы увеличил количество данных и объем словарей.

---

## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/Lol547/BPE-transformer-translation-eng-rus.git
cd BPE-transformer-translation-eng-rus
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

**Требования:**
- Python 3.8+
- PyTorch (с поддержкой CUDA, если есть GPU)
- torchvision, numpy, matplotlib, tqdm, nltk

---

## Использование

### Инференс

**Пример перевода одного предложения:**

```python
from src.translate import load_model, generate_translation

# Загрузка модели и токенизаторов
model, eng_tokenizer, rus_tokenizer = load_model(
    checkpoint_path="checkpoints/best_model.pt",
    eng_tokenizer_path="checkpoints/english_bpe_tokenizer.pkl",
    rus_tokenizer_path="checkpoints/russian_bpe_tokenizer.pkl",
    device="cuda"  # или "cpu"
)

# Перевод
result = generate_translation(
    model, eng_tokenizer, rus_tokenizer,
    "How are you today?",
    max_len=96,
    device="cuda"
)
print(result)  # как вы сегодня
```

**Командная строка:**

```bash
python src/translate.py --text "Hello world!" --weights checkpoints/best_model.pt --eng_tokenizer checkpoints/english_bpe_tokenizer.pkl --rus_tokenizer checkpoints/russian_bpe_tokenizer.pkl
```

### Обучение

```bash
python src/train.py --epochs 40 --batch_size 64 --hidden_dim 384 --num_layers 5 --num_heads 8
```

Все гиперпараметры можно настроить через аргументы командной строки или изменить в конфигурационном файле.

---

## Структура проекта

```
transformer-translator/
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── transformer_translator.ipynb    # Исходный ноутбук с экспериментами
│
├── src/
│   ├── __init__.py
│   ├── model.py                         # TransformerModel, PositionalEmbedding, Encoder/Decoder слои
│   ├── tokenizer.py                     # BPE-токенизация (SubWordTokenizer)
│   ├── dataset.py                       # TranslationDataset, make_dataset
│   ├── utils.py                         # NoamScheduler, EarlyStopping, генерация перевода
│   ├── train.py                         # Скрипт обучения
│   └── translate.py                     # Скрипт инференса
│
├── configs/
│   └── default.yaml                     # Конфигурация гиперпараметров
│
├── checkpoints/                         # Папка для весов (в .gitignore)
│   ├── best_model.pt
│   ├── english_bpe_tokenizer.pkl
│   └── russian_bpe_tokenizer.pkl
│
└── data/                                # Датасет (скачивается автоматически)
    └── rus-eng.zip
```

---

## Веса модели

Для работы необходимы следующие файлы:
1. **`best_seq2seq_model_transformer.pt`** - веса модели (полный чекпоинт с config).
2. **`english_bpe_tokenizer.pkl`** - BPE-токенизатор для английского языка.
3. **`russian_bpe_tokenizer.pkl`** - BPE-токенизатор для русского языка.

**Ссылки для скачивания:**
[Веса и BPE-токенизаторы](https://drive.google.com/drive/folders/1zE077T69BIF5fhroV4O-tw6jlar-Wdcp?usp=sharing)



## Контакты

По вопросам или найденным ошибкам пишите: sokolovkirill489@gmail.com TG: @qqkiru

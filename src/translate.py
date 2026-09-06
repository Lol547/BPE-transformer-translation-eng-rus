import torch
import pickle
import argparse
from utils import generate_translation, load_model


def main():
    parser = argparse.ArgumentParser(description="Перевод с английского на русский")
    parser.add_argument("--text", type=str, required=True, help="Текст на английском")
    parser.add_argument("--weights", type=str, default="best_transformer_model.pt", help="Путь к весам модели")
    parser.add_argument("--eng_tokenizer", type=str, default="english_bpe_tokenizer.pkl", help="Путь к токенизатору EN")
    parser.add_argument("--rus_tokenizer", type=str, default="russian_bpe_tokenizer.pkl", help="Путь к токенизатору RU")
    parser.add_argument("--max_len", type=int, default=96, help="Максимальная длина перевода")
    parser.add_argument("--device", type=str, default="cuda", help="cuda или cpu")
    args = parser.parse_args()
    
    model, eng_tokenizer, rus_tokenizer = load_model(
        args.weights, args.eng_tokenizer, args.rus_tokenizer, args.device
    )
    
    translation = generate_translation(
        model,
        eng_tokenizer,
        rus_tokenizer,
        args.text,
        max_len=args.max_len,
        device=args.device,
    )
    
    print(f"Исходный текст: {args.text}")
    print(f"Перевод: {translation}")


if __name__ == "__main__":
    main()

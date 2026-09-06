import os
import random
import re
import pathlib
import urllib.request
import zipfile
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import pickle

from model import TransformerModel
from tokenizer import compute_sub_word_vocabulary, SubWordTokenizer
from dataset import make_dataset
from utils import NoamScheduler, EarlyStopping


def download_and_prepare_data():
    """Скачивает и распаковывает датасет rus-eng."""
    zip_path = pathlib.Path("rus-eng.zip")
    extract_dir = pathlib.Path("rus-eng-data")
    text_path = extract_dir / "rus.txt"
    
    if not text_path.exists():
        if not zip_path.exists():
            print("Скачивание архива...")
            url = "https://www.manythings.org/anki/rus-eng.zip"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            )
            with urllib.request.urlopen(req, timeout=15) as response, open(zip_path, "wb") as out_file:
                chunk_size = 1024 * 1024
                downloaded = 0
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    print(f"Загружено: {downloaded / (1024 * 1024):.1f} MB", end="\r")
            print("\nСкачивание завершено")
        
        print("Распаковка архива...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Распаковка завершена")
    
    return text_path


def load_data(text_path):
    """Загружает пары предложений из файла."""
    with open(text_path, encoding="utf-8") as f:
        lines = f.read().split("\n")[:-1]
    
    text_pairs = []
    for line in lines:
        parts = line.split("\t")
        if len(parts) >= 2:
            english, russian = parts[0], parts[1]
            text_pairs.append((english, russian))
    return text_pairs


def main():
    parser = argparse.ArgumentParser(description="Обучение Transformer переводчика")
    parser.add_argument("--epochs", type=int, default=40, help="Количество эпох")
    parser.add_argument("--batch_size", type=int, default=64, help="Размер батча")
    parser.add_argument("--sequence_length", type=int, default=96, help="Максимальная длина последовательности")
    parser.add_argument("--vocab_size", type=int, default=30000, help="Размер BPE-словаря")
    parser.add_argument("--hidden_dim", type=int, default=384, help="Размер скрытого слоя")
    parser.add_argument("--intermediate_dim", type=int, default=1024, help="Размер промежуточного слоя FFN")
    parser.add_argument("--num_heads", type=int, default=8, help="Количество голов внимания")
    parser.add_argument("--num_layers", type=int, default=5, help="Количество слоёв энкодера/декодера")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout")
    parser.add_argument("--warmup_steps", type=int, default=10000, help="Warmup steps для Noam-планировщика")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")
    
    text_path = download_and_prepare_data()
    text_pairs = load_data(text_path)
    random.shuffle(text_pairs)
    
    train_len = int(0.85 * len(text_pairs))
    val_len = int(0.075 * len(text_pairs))
    train_pairs = text_pairs[:train_len]
    val_pairs = text_pairs[train_len:train_len + val_len]
    test_pairs = text_pairs[train_len + val_len:]
    
    print(f"Всего пар: {len(text_pairs)}")
    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")
    
    print("Создание BPE словарей...")
    eng_vocab, eng_merges = compute_sub_word_vocabulary(
        dataset=[eng for eng, _ in train_pairs],
        vocab_size=args.vocab_size
    )
    rus_vocab, rus_merges = compute_sub_word_vocabulary(
        dataset=[rus for _, rus in train_pairs],
        vocab_size=args.vocab_size
    )
    
    eng_tokenizer = SubWordTokenizer(eng_vocab, eng_merges)
    rus_tokenizer = SubWordTokenizer(rus_vocab, rus_merges)
    
    src_vocab_size = len(eng_tokenizer.vocabulary)
    tgt_vocab_size = len(rus_tokenizer.vocabulary)
    print(f"Размер словаря EN: {src_vocab_size}, RU: {tgt_vocab_size}")
    
    train_ds = make_dataset(
        train_pairs, eng_tokenizer, rus_tokenizer, args.sequence_length,
        args.batch_size, shuffle=True
    )
    val_ds = make_dataset(
        val_pairs, eng_tokenizer, rus_tokenizer, args.sequence_length,
        args.batch_size, shuffle=False
    )
    
    model = TransformerModel(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        sequence_length=args.sequence_length,
        hidden_dim=args.hidden_dim,
        intermediate_dim=args.intermediate_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Всего параметров: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model=args.hidden_dim, warmup_steps=args.warmup_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    
    history_train_loss = []
    history_train_acc = []
    history_val_loss = []
    history_val_acc = []
    
    best_val_loss = float("inf")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_ds, desc=f"Эпоха {epoch:02d}/{args.epochs:02d} [Train]")
        for batch_features, batch_labels in pbar:
            source = batch_features["english"].to(device)
            target = batch_features["russian"].to(device)
            labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(source, target)
                loss = criterion(outputs.reshape(-1, tgt_vocab_size), labels.reshape(-1))
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            
            train_loss += loss.item()
            
            mask = labels != 0
            preds = torch.argmax(outputs, dim=-1)
            train_correct += ((preds == labels) & mask).sum().item()
            train_total += mask.sum().item()
            
            current_acc = (train_correct / train_total * 100) if train_total > 0 else 0.0
            current_lr = optimizer.param_groups[0]["lr"]
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{current_acc:.1f}%",
                "lr": f"{current_lr:.6f}"
            })
        
        avg_train_loss = train_loss / len(train_ds)
        train_accuracy = (train_correct / train_total * 100) if train_total > 0 else 0.0
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                for batch_features, batch_labels in tqdm(val_ds, desc=f"Эпоха {epoch:02d}/{args.epochs:02d} [Val]"):
                    source = batch_features["english"].to(device)
                    target = batch_features["russian"].to(device)
                    labels = batch_labels.to(device)
                    
                    outputs = model(source, target)
                    loss = criterion(outputs.reshape(-1, tgt_vocab_size), labels.reshape(-1))
                    
                    val_loss += loss.item()
                    
                    mask = labels != 0
                    preds = torch.argmax(outputs, dim=-1)
                    val_correct += ((preds == labels) & mask).sum().item()
                    val_total += mask.sum().item()
        
        avg_val_loss = val_loss / len(val_ds)
        val_accuracy = (val_correct / val_total * 100) if val_total > 0 else 0.0
        
        history_train_loss.append(avg_train_loss)
        history_train_acc.append(train_accuracy)
        history_val_loss.append(avg_val_loss)
        history_val_acc.append(val_accuracy)
        
        print(
            f"Итог эпохи {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}% | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_accuracy": val_accuracy,
                "val_loss": avg_val_loss,
                "history_train_loss": history_train_loss,
                "history_train_acc": history_train_acc,
                "history_val_loss": history_val_loss,
                "history_val_acc": history_val_acc,
                "config": {
                    "src_vocab_size": src_vocab_size,
                    "tgt_vocab_size": tgt_vocab_size,
                    "sequence_length": args.sequence_length,
                    "hidden_dim": args.hidden_dim,
                    "intermediate_dim": args.intermediate_dim,
                    "num_heads": args.num_heads,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                }
            }
            torch.save(checkpoint, "best_transformer_model.pt")
            print(f"Чекпоинт сохранён (Val Loss: {avg_val_loss:.4f})")
    
    with open("english_bpe_tokenizer.pkl", "wb") as f:
        pickle.dump({
            "vocabulary": eng_tokenizer.vocabulary,
            "merges": eng_tokenizer.merges
        }, f)
    
    with open("russian_bpe_tokenizer.pkl", "wb") as f:
        pickle.dump({
            "vocabulary": rus_tokenizer.vocabulary,
            "merges": rus_tokenizer.merges
        }, f)
    
    print("Модель и токенизаторы сохранены.")


if __name__ == "__main__":
    main()

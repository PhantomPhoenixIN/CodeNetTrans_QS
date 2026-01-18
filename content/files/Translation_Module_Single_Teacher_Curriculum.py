#!/usr/bin/env python3
"""
FULL fine-tuning of CodeT5 for multi-language → Java translation
with teacher-aware cleaned references.

Curriculum training: S1 → S2 → S3
Teacher-isolated training (NO cross-teacher mixing)

Differences from partial FT:
- All parameters trainable
- Early stopping based ONLY on validation loss
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
)

from accelerate import Accelerator

# =====================================================
# ================= USER CONFIG =======================
# =====================================================

TARGET_SCORE   = 3                  # 1 → 2 → 3
TARGET_TEACHER = "DeepSeekCoder"    # STRICT teacher isolation

# Progressive model loading
MODEL_DIR = "/home/swaminathanj/TransRectify_New/models/CodeT5_Translation_Deepseek/score_2_deepseekcoder/best"

DATASET_PATH = "/home/swaminathanj/TransRectify_New/content/datasets/codenet_combined_translator_dataset_astfixed.jsonl"

OUTPUT_ROOT = (
    "/home/swaminathanj/TransRectify_New/models/"
    "CodeT5_Translation_Deepseek"
)

PROMPT_MAX_LEN = 512
TARGET_MAX_LEN = 512

TRAIN_BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 1

LEARNING_RATE = 5e-5
NUM_EPOCHS = 30
WARMUP_STEPS = 100

EARLY_STOP_PATIENCE = 3
EARLY_STOP_DELTA = 1e-4

SEED = 42
USE_FP16 = True

# =====================================================

PROMPT_TEMPLATE = """### Task:
Translate the following {input_language} program into Java.
Preserve semantics and structure.
Output only valid Java source code.

### Source Code:
{source_code}

### Java Code:
"""

# =====================================================
# ========== TEACHER-AWARE REFERENCE ==================
# =====================================================

def get_reference_java(obj):
    teacher = obj.get("teacher")

    if teacher == "StarCoder":
        return obj.get("translated_java_code")
    if teacher == "QwenCoder":
        return obj.get("qwencoder_translation_clean")
    if teacher == "DeepSeekCoder":
        return obj.get("deepseekcoder_translation_clean")

    return None

# =====================================================
# ================= DATA LOADING ======================
# =====================================================

def load_translation_jsonl(path, split, score, seed):
    rng = random.Random(seed)
    records = []
    teacher_count = {}

    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            if not ln.strip():
                continue
            try:
                obj = json.loads(ln)
            except Exception:
                continue

            if obj.get("split") != split:
                continue
            if obj.get("score") != score:
                continue
            if obj.get("output_language") != "Java":
                continue

            teacher = obj.get("teacher", "UNKNOWN")

            # 🔒 STRICT teacher isolation
            if teacher != TARGET_TEACHER:
                continue

            src = obj.get("source_code")
            tgt = get_reference_java(obj)
            lang = obj.get("input_language")

            if not isinstance(src, str) or not isinstance(tgt, str):
                continue

            src, tgt = src.strip(), tgt.strip()
            if not src or not tgt:
                continue

            records.append({
                "input_language": lang,
                "source": src,
                "target": tgt,
                "teacher": teacher,
            })

            teacher_count[teacher] = teacher_count.get(teacher, 0) + 1

    rng.shuffle(records)

    print(
        f"\n{split.upper()} SPLIT | "
        f"Score={score} | Teacher={TARGET_TEACHER}"
    )
    print("-" * 60)
    print(f"Total instances: {len(records)}")
    for t, c in teacher_count.items():
        print(f"  {t:15s}: {c}")
    print("-" * 60)

    return records

# =====================================================
# ================= DATASET ===========================
# =====================================================

class CodeTranslationDataset(Dataset):
    def __init__(self, records, tokenizer):
        self.records = records
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]

        prompt = PROMPT_TEMPLATE.format(
            input_language=r["input_language"],
            source_code=r["source"]
        )

        enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=PROMPT_MAX_LEN,
        )

        dec = self.tokenizer(
            text=r["target"],
            truncation=True,
            max_length=TARGET_MAX_LEN,
        )

        labels = dec["input_ids"]
        if not labels:
            labels = [self.tokenizer.eos_token_id]

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }

def collate_fn(batch, pad_id):
    max_in = max(len(b["input_ids"]) for b in batch)
    max_lb = max(len(b["labels"]) for b in batch)

    input_ids, attn, labels = [], [], []

    for b in batch:
        input_ids.append(b["input_ids"] + [pad_id] * (max_in - len(b["input_ids"])))
        attn.append(b["attention_mask"] + [0] * (max_in - len(b["attention_mask"])))

        lab = b["labels"] + [pad_id] * (max_lb - len(b["labels"]))
        lab = [(-100 if t == pad_id else t) for t in lab]
        labels.append(lab)

    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attn),
        "labels": torch.tensor(labels),
    }

# =====================================================
# ================= VALIDATION ========================
# =====================================================

def evaluate_loss(model, dataloader):
    model.eval()
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for batch in dataloader:
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            steps += 1

    return total_loss / max(steps, 1)

# =====================================================
# ================= MAIN ==============================
# =====================================================

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    accelerator = Accelerator(
        mixed_precision="fp16" if USE_FP16 else None
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    model.resize_token_embeddings(len(tokenizer))

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    print(
        f"\n=== CURRICULUM FINETUNING | "
        f"Score={TARGET_SCORE} | Teacher={TARGET_TEACHER} ==="
    )

    train_records = load_translation_jsonl(
        DATASET_PATH, "train", TARGET_SCORE, SEED
    )
    val_records = load_translation_jsonl(
        DATASET_PATH, "val", TARGET_SCORE, SEED + 1
    )

    train_ds = CodeTranslationDataset(train_records, tokenizer)
    val_ds   = CodeTranslationDataset(val_records, tokenizer)

    train_dl = DataLoader(
        train_ds,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    model, optimizer, train_dl, val_dl = accelerator.prepare(
        model, optimizer, train_dl, val_dl
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        WARMUP_STEPS,
        NUM_EPOCHS * len(train_dl)
    )

    out_dir = os.path.join(
        OUTPUT_ROOT,
        f"score_{TARGET_SCORE}_{TARGET_TEACHER.lower()}"
    )
    os.makedirs(out_dir, exist_ok=True)

    best_val_loss = float("inf")
    no_improve = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch in train_dl:
            outputs = model(**batch)
            loss = outputs.loss / GRAD_ACCUM_STEPS
            accelerator.backward(loss)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        val_loss = evaluate_loss(model, val_dl)

        print(
            f"[Epoch {epoch}] "
            f"Train Loss={total_loss / len(train_dl):.6f} | "
            f"Val Loss={val_loss:.6f}"
        )

        if val_loss < best_val_loss - EARLY_STOP_DELTA:
            best_val_loss = val_loss
            no_improve = 0
            accelerator.unwrap_model(model).save_pretrained(
                os.path.join(out_dir, "best")
            )
            tokenizer.save_pretrained(os.path.join(out_dir, "best"))
        else:
            no_improve += 1

        if no_improve >= EARLY_STOP_PATIENCE:
            print("Early stopping triggered")
            break

if __name__ == "__main__":
    main()
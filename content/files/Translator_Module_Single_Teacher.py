#!/usr/bin/env python3
"""
Partial fine-tuning of CodeT5 for multi-language → Java translation
using teacher-aware, score-isolated references.

This version:
- trains ONLY on DeepSeekCoder data
- isolates a single score bucket
- is safe for HPC, fp16, gradient checkpointing
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
import importlib.util

# =====================================================
# ================= USER CONFIG =======================
# =====================================================

TARGET_SCORE   = 1                  # 1 / 2 / 3
TARGET_TEACHER = "DeepSeekCoder"    # STRICT teacher isolation

MODEL_DIR = "/home/swaminathanj/CodeFixNet/content/models/CodeT5/CodeT5"
DATASET_PATH = "/home/swaminathanj/TransRectify_New/content/datasets/codenet_combined_translator_dataset_astfixed.jsonl"
EVAL_METRICS_PATH = "/home/swaminathanj/CodeFixNet/content/files/eval_metrics.py"

OUTPUT_ROOT = "/home/swaminathanj/TransRectify_New/models/CodeT5_Translation_Deepseek"

PROMPT_MAX_LEN = 512
TARGET_MAX_LEN = 512

TRAIN_BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 1

LEARNING_RATE = 2e-4
NUM_EPOCHS = 30
WARMUP_STEPS = 100
MAX_GRAD_NORM = 1.0

UNFREEZE_LAST_K = 1
EARLY_STOP_PATIENCE = 3
EARLY_STOP_DELTA = 0.001

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
# ========== TEACHER-AWARE REFERENCE SELECTOR =========
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
            if TARGET_TEACHER is not None and teacher != TARGET_TEACHER:
                continue

            src = obj.get("source_code")
            tgt = get_reference_java(obj)
            lang = obj.get("input_language")

            if not isinstance(src, str) or not isinstance(tgt, str):
                continue

            src = src.strip()
            tgt = tgt.strip()

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

    print(f"\n{split.upper()} SPLIT STATISTICS | Score={score} | Teacher={TARGET_TEACHER}")
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
# ================= MODEL UTILS =======================
# =====================================================

def unfreeze_last_k_decoder_blocks(model, k):
    for p in model.parameters():
        p.requires_grad = False

    block_ids = sorted({
        int(n.split(".")[2])
        for n, _ in model.named_modules()
        if n.startswith("decoder.block.") and n.split(".")[2].isdigit()
    })

    for i in block_ids[-k:]:
        for n, p in model.named_parameters():
            if f"decoder.block.{i}." in n:
                p.requires_grad = True

    for n, p in model.named_parameters():
        if "lm_head" in n:
            p.requires_grad = True

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"Trainable parameter tensors: {len(trainable)}")

# =====================================================
# ================= EVALUATION ========================
# =====================================================

def evaluate(model, tokenizer, records, device, max_eval=256):
    model.eval()
    preds, refs = [], []

    for r in records[:max_eval]:
        prompt = PROMPT_TEMPLATE.format(
            input_language=r["input_language"],
            source_code=r["source"]
        )

        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=PROMPT_MAX_LEN
        ).to(device)

        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=128,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        preds.append(tokenizer.decode(gen[0], skip_special_tokens=True))
        refs.append(r["target"])

    bleu, codebleu = [], []
    for p, r in zip(preds, refs):
        b, c = eval_metrics.compute_scores(p, r)
        bleu.append(b)
        codebleu.append(c)

    return sum(bleu) / len(bleu), sum(codebleu) / len(codebleu)

# =====================================================
# ================= MAIN ==============================
# =====================================================

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    spec = importlib.util.spec_from_file_location("eval_metrics", EVAL_METRICS_PATH)
    global eval_metrics
    eval_metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_metrics)

    accelerator = Accelerator(mixed_precision="fp16" if USE_FP16 else None)
    device = accelerator.device

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    model.resize_token_embeddings(len(tokenizer))

    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    unfreeze_last_k_decoder_blocks(model, UNFREEZE_LAST_K)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE
    )

    print(f"\n=== TRAINING CodeT5 | Score={TARGET_SCORE} | Teacher={TARGET_TEACHER} ===")

    train_records = load_translation_jsonl(DATASET_PATH, "train", TARGET_SCORE, SEED)
    val_records   = load_translation_jsonl(DATASET_PATH, "val",   TARGET_SCORE, SEED + 1)
    test_records  = load_translation_jsonl(DATASET_PATH, "test",  TARGET_SCORE, SEED + 2)

    train_ds = CodeTranslationDataset(train_records, tokenizer)
    train_dl = DataLoader(
        train_ds,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    model, optimizer, train_dl = accelerator.prepare(model, optimizer, train_dl)

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

    best_cb = -1e9
    no_improve = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch in train_dl:
            out = model(**batch)
            loss = out.loss / GRAD_ACCUM_STEPS
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        bleu, codebleu = evaluate(model, tokenizer, val_records, device)
        print(f"[Epoch {epoch}] VAL BLEU={bleu:.4f} CodeBLEU={codebleu:.4f}")

        if codebleu > best_cb + EARLY_STOP_DELTA:
            best_cb = codebleu
            no_improve = 0
            accelerator.unwrap_model(model).save_pretrained(os.path.join(out_dir, "best"))
            tokenizer.save_pretrained(os.path.join(out_dir, "best"))
        else:
            no_improve += 1

        if no_improve >= EARLY_STOP_PATIENCE:
            print("Early stopping triggered")
            break

    print("\nFinal TEST evaluation")
    best_model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(out_dir, "best")).to(device)
    bleu, codebleu = evaluate(best_model, tokenizer, test_records, device)
    print(f"[TEST] BLEU={bleu:.4f} CodeBLEU={codebleu:.4f}")

if __name__ == "__main__":
    main()
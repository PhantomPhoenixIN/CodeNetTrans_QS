#!/usr/bin/env python3
"""
Generate Java translations on TEST set using trained CodeT5 models
(score = 1 / 2 / 3) and store outputs as JSONL files.
"""

import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# =====================================================
# ================= CONFIG ============================
# =====================================================

DATASET_PATH = "/home/swaminathanj/TransRectify_New/content/datasets/codenet_combined_translator_dataset_astfixed.jsonl"
MODEL_ROOT   = "/home/swaminathanj/TransRectify_New/models/CodeT5_Translation"
OUTPUT_DIR   = "/home/swaminathanj/TransRectify_New/content/datasets/predictions"

PROMPT_MAX_LEN = 512
GEN_MAX_TOKENS = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SCORES = [2, 3]

PROMPT_TEMPLATE = """### Task:
Translate the following {input_language} program into Java.
Preserve semantics and structure.
Output only valid Java source code.

### Source Code:
{source_code}

### Java Code:
"""

os.makedirs(OUTPUT_DIR, exist_ok=True)

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
# ========== LOAD TEST RECORDS ========================
# =====================================================

def load_test_records(score):
    records = []

    with open(DATASET_PATH, "r", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)

            if obj.get("split") != "test":
                continue
            if obj.get("score") != score:
                continue
            if obj.get("output_language") != "Java":
                continue

            src = obj.get("source_code")
            tgt = get_reference_java(obj)

            if not isinstance(src, str) or not isinstance(tgt, str):
                continue

            records.append({
                "problem_id": obj.get("problem_id"),
                "input_language": obj.get("input_language"),
                "teacher": obj.get("teacher"),
                "score": score,
                "source_code": src.strip(),
                "reference_java": tgt.strip(),
            })

    print(f"[Score {score}] Loaded {len(records)} TEST instances")
    return records

# =====================================================
# ========== GENERATION ===============================
# =====================================================

def generate_predictions(score):
    model_dir = os.path.join(MODEL_ROOT, f"score_{score}", "best")
    out_path  = os.path.join(
        OUTPUT_DIR,
        f"codenet_test_score{score}_model_translation.jsonl"
    )

    print(f"\n=== Generating TEST translations | SCORE = {score} ===")
    print(f"Model: {model_dir}")
    print(f"Output: {out_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(DEVICE)
    model.eval()

    records = load_test_records(score)

    with open(out_path, "w", encoding="utf-8") as fout:
        for idx, r in enumerate(tqdm(records, desc=f"Score {score} inference"), start=1):
            prompt = PROMPT_TEMPLATE.format(
                input_language=r["input_language"],
                source_code=r["source_code"]
            )

            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=PROMPT_MAX_LEN
            ).to(DEVICE)

            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    max_new_tokens=GEN_MAX_TOKENS,
                    do_sample=False,
                    num_beams=1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                )

            pred = tokenizer.decode(gen[0], skip_special_tokens=True).strip()

            out_obj = {
                **r,
                "predicted_java": pred
            }

            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

            # 🔥 FLUSH EVERY 10 INSTANCES
            if idx % 10 == 0:
                fout.flush()

    print(f"✔ Finished SCORE {score}")

# =====================================================
# ================= MAIN ==============================
# =====================================================

def main():
    for score in SCORES:
        generate_predictions(score)

    print("\nAll test translations generated successfully.")

if __name__ == "__main__":
    main()
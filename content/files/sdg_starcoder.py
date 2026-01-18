#!/usr/bin/env python3
"""
StarCoder → Java synthetic translation generator

Adds:
  - starcoder_translation_raw
  - starcoder_translation_clean

Guarantees:
  - Row order preserved
  - Strict schema
  - Periodic flushing
  - Safe truncation (no OverflowError)
"""

import json
import time
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------
MODEL_PATH = "/home/swaminathanj/TransRectify_New/content/models/StarCoder_new"
INPUT_JSONL = "/home/swaminathanj/TransRectify_New/content/datasets/codenet_single_solution.jsonl"
OUTPUT_JSONL = "/home/swaminathanj/TransRectify_New/content/datasets/codenet_single_solution_starcoder.jsonl"

# ------------------------------------------------------------------
# GENERATION SETTINGS
# ------------------------------------------------------------------
MAX_INPUT_TOKENS = 4096        # IMPORTANT: prevents OverflowError
MAX_NEW_TOKENS = 512
DO_SAMPLE = False
FLUSH_EVERY = 10
SLEEP_BETWEEN = 0.01

# ------------------------------------------------------------------
# PROMPT
# ------------------------------------------------------------------
def build_prompt(source_code: str, source_lang: str) -> str:
    return (
        f"### Translate the following {source_lang} code to Java.\n\n"
        f"{source_code}\n\n"
        f"### Java translation:\n"
    )

# ------------------------------------------------------------------
# CLEANING
# ------------------------------------------------------------------
def extract_java_clean(text: str) -> str:
    """
    Extract Java code block if present, else return trimmed output.
    """
    if "```java" in text:
        return text.split("```java", 1)[1].split("```", 1)[0].strip()
    if "```" in text:
        return text.split("```", 1)[1].strip()
    return text.strip()

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[INFO] Loading StarCoder tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=True
    )

    # StarCoder has no pad token
    tokenizer.pad_token = tokenizer.eos_token

    print("[INFO] Loading StarCoder model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)

    model.eval()

    buffer = []
    processed = 0

    print("[INFO] Starting StarCoder translation pass...")

    with open(INPUT_JSONL, "r", encoding="utf-8") as fin, \
         open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="StarCoder → Java"):
            row = json.loads(line)

            src = row.get("source_code", "")
            lang = row.get("input_language", "Unknown")

            if not src:
                out_row = dict(row)
                out_row["starcoder_translation_raw"] = None
                out_row["starcoder_translation_clean"] = None
                buffer.append(out_row)
                continue

            prompt = build_prompt(src, lang)

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_INPUT_TOKENS,
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=DO_SAMPLE,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            decoded = tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            raw_translation = decoded
            clean_translation = extract_java_clean(decoded)

            # STRICT ROW PRESERVATION
            out_row = dict(row)
            out_row["starcoder_translation_raw"] = raw_translation
            out_row["starcoder_translation_clean"] = clean_translation

            buffer.append(out_row)
            processed += 1

            # ------------------------------------------------------
            # PERIODIC FLUSH
            # ------------------------------------------------------
            if processed % FLUSH_EVERY == 0:
                for r in buffer:
                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                fout.flush()
                buffer.clear()

            time.sleep(SLEEP_BETWEEN)

        # Final flush
        if buffer:
            for r in buffer:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            fout.flush()

    print("[DONE] StarCoder translations written to:")
    print("       ", OUTPUT_JSONL)


if __name__ == "__main__":
    main()
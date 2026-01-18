#!/usr/bin/env python3
"""
Qwen2.5-Coder → Java synthetic translation generator

Outputs TWO fields:
  1) qwencoder_translation_raw
  2) qwencoder_translation_clean

Safe for HPC:
  - periodic flushing
  - deterministic decoding
"""

import json
import time
import torch
import os
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

# ------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------
MODEL_PATH = "/home/swaminathanj/TransRectify_New/content/models/QwenCoder_new"
INPUT_JSONL = "/home/swaminathanj/TransRectify_New/content/datasets/codenet_single_solution.jsonl"
OUTPUT_JSONL = "/home/swaminathanj/TransRectify_New/content/datasets/codenet_single_solution_qwencoder_2.jsonl"

# ------------------------------------------------------------------
# GENERATION SETTINGS
# ------------------------------------------------------------------
MAX_NEW_TOKENS = 512
DO_SAMPLE = False          # deterministic
NUM_BEAMS = 1
SLEEP_BETWEEN = 0.01
FLUSH_EVERY = 10           # HPC-safe flush frequency

# ------------------------------------------------------------------
# PROMPT
# ------------------------------------------------------------------
def build_prompt(source_code: str, source_lang: str) -> str:
    return (
        f"### Translate the following {source_lang} code to Java.\n\n"
        f"{source_code}\n\n"
        f"### Java translation:\n"
    )

def extract_java(decoded: str, prompt: str) -> str:
    if prompt in decoded:
        return decoded.split(prompt, 1)[1].strip()
    return decoded.strip()

# ------------------------------------------------------------------
# OUTPUT SANITIZER
# ------------------------------------------------------------------
def clean_java_output(text: str) -> str:
    """
    Remove explanations, chat artifacts, markdown fences.
    Keep only Java code.
    """
    if not text:
        return None

    # Remove Human / Assistant turns
    text = re.split(r"\bHuman:|\bAssistant:", text)[0]

    # Extract ```java ... ``` if present
    match = re.search(r"```java(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Remove any remaining markdown fences
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

    # Heuristic: start from first 'class'
    idx = text.find("class ")
    if idx != -1:
        return text[idx:].strip()

    return text.strip()

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[INFO] Loading Qwen tokenizer...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[INFO] Loading Qwen2.5-Coder model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)

    model.eval()

    print("[INFO] Starting QwenCoder → Java translation pass...")

    with open(INPUT_JSONL, "r", encoding="utf-8") as fin, \
         open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:

        for idx, line in enumerate(tqdm(fin, desc="QwenCoder → Java"), start=1):
            entry = json.loads(line)

            src = entry.get("source_code", "")
            lang = entry.get("input_language", "Unknown")

            if not src:
                entry["qwencoder_translation_raw"] = None
                entry["qwencoder_translation_clean"] = None
            else:
                prompt = build_prompt(src, lang)

                tokens = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                )

                input_ids = tokens["input_ids"].to(device)
                attention_mask = tokens.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=DO_SAMPLE,
                        num_beams=NUM_BEAMS,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        temperature=None,
                        top_p=None,
                        top_k=None,
                    )

                decoded = tokenizer.decode(
                    output_ids[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )

                raw = extract_java(decoded, prompt)
                clean = clean_java_output(raw)

                entry["qwencoder_translation_raw"] = raw
                entry["qwencoder_translation_clean"] = clean

            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

            # ---------------- periodic flush ----------------
            if idx % FLUSH_EVERY == 0:
                fout.flush()
                os.fsync(fout.fileno())
            # -------------------------------------------------

            time.sleep(SLEEP_BETWEEN)

        # Final flush
        fout.flush()
        os.fsync(fout.fileno())

    print("[DONE] QwenCoder synthetic translations written to:")
    print("       ", OUTPUT_JSONL)

if __name__ == "__main__":
    main()
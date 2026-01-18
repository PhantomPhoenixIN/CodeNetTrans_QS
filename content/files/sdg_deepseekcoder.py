#!/usr/bin/env python3
"""
DeepSeekCoder → Java code translation
- Uses local DeepSeekCoder model
- Preserves full row
- Stores raw + cleaned translations
- Periodic flushing (every N samples)
"""

import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
MODEL_DIR = "/home/swaminathanj/TransRectify_New/content/models/DeepseekCoder"

INPUT_JSONL = INPUT_JSONL = "/home/swaminathanj/TransRectify_New/content/datasets/codenet_single_solution.jsonl"
OUTPUT_JSONL = "/home/swaminathanj/TransRectify_New/content/datasets/codenet_single_solution_deepseekcoder.jsonl"

FLUSH_EVERY = 10
MAX_NEW_TOKENS = 512


# ------------------------------------------------------------------
# Prompt builder
# ------------------------------------------------------------------
def build_prompt(src_code: str, src_lang: str, tgt_lang: str = "Java") -> str:
    return (
        f"### Translate the following {src_lang} code to {tgt_lang}.\n\n"
        f"{src_code}\n\n"
        f"### {tgt_lang} translation:\n"
    )


# ------------------------------------------------------------------
# Very light cleanup (NO strict filtering)
# ------------------------------------------------------------------
def clean_generation(text: str) -> str:
    """
    Minimal cleanup:
    - Remove markdown fences
    - Do NOT remove explanations aggressively
    """
    lines = []
    for line in text.splitlines():
        if line.strip().startswith("```"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


# ------------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    print("[INFO] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    ).eval()

    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)

    buffer = []
    count = 0

    with open(INPUT_JSONL, "r", encoding="utf-8") as fin, \
         open(OUTPUT_JSONL, "a", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="DeepSeekCoder → Java"):
            row = json.loads(line)

            prompt = build_prompt(
                row["source_code"],
                row["input_language"],
                row["output_language"]
            )

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False
                )

            decoded = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True
            )

            row["deepseekcoder_translation_raw"] = decoded
            row["deepseekcoder_translation_clean"] = clean_generation(decoded)

            buffer.append(row)
            count += 1

            # ------------------------------------------------------
            # Periodic flushing
            # ------------------------------------------------------
            if count % FLUSH_EVERY == 0:
                for r in buffer:
                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                fout.flush()
                buffer.clear()

        # ----------------------------------------------------------
        # Final flush
        # ----------------------------------------------------------
        for r in buffer:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
        fout.flush()

    print("[DONE] DeepSeekCoder translations completed.")


# ------------------------------------------------------------------
if __name__ == "__main__":
    main()

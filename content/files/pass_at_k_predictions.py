#!/usr/bin/env python3
"""
PREDICTION-ONLY SCRIPT (FINAL, IMPORT-SAFE, FLUSHED, CUDA-CLEAN)

For each model:
- Uses FULL deduplicated test set (1308 instances)
- Generate 1 greedy Java solution
- Generate N stochastic Java solutions
- Strip Java based on MODEL TYPE
- Prepend safe Java imports
- Flush JSONL after EVERY instance
- Print progress after EVERY instance
- Clean CUDA after each model
"""

# =====================================================
# IMPORTS
# =====================================================

import os
import re
import json
import gc
import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM
)

# =====================================================
# CONFIG
# =====================================================

DATASET_PATH = (
    "/home/swaminathanj/TransRectify_New/content/datasets/"
    "codenet_combined_translator_dataset_astfixed.jsonl"
)

MODEL_PATHS = [
    # "/home/swaminathanj/TransRectify_New/content/models/StarCoder_new",
    # "/home/swaminathanj/TransRectify_New/content/models/QwenCoder_new",
    # "/home/swaminathanj/TransRectify_New/content/models/DeepseekCoder",
    # "/home/swaminathanj/TransRectify_New/models/pass_at_k_eval_models/FFT_S3",
    # "/home/swaminathanj/TransRectify_New/models/pass_at_k_eval_models/Curr_Final",
    # "/home/swaminathanj/TransRectify_New/models/pass_at_k_eval_models/FFT_S1",
    # "/home/swaminathanj/TransRectify_New/models/pass_at_k_eval_models/FFT_S2",
    # "/home/swaminathanj/TransRectify_New/models/pass_at_k_eval_models/Curr_S2",
    # "/home/swaminathanj/TransRectify_New/models/pass_at_k_eval_models/Curr_Final_Rev",
    # "/home/swaminathanj/TransRectify_New/models/pass_at_k_eval_models/Curr_Final_Deepseek",
]

OUTPUT_DIR = (
    "/home/swaminathanj/TransRectify_New/content/datasets/pass_at_k_predictions"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_CANDIDATES = 5
MAX_GEN_LEN = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPT_TEMPLATE = """### Task:
Translate the following {input_language} program into Java.
Preserve semantics and structure.
Output only valid Java source code.

### Source Code:
{source_code}

### Java Code:
"""

DEFAULT_IMPORTS = """import java.io.*;
import java.util.*;

"""

# =====================================================
# LOAD FULL DEDUPLICATED TEST SET (1308 GUARANTEED)
# =====================================================

def load_test_samples_dedup():
    """
    Load ENTIRE test set and deduplicate by
    (problem_id, input_language).

    Must produce EXACTLY 1308 instances.
    """
    unique = {}

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue

            obj = json.loads(ln)

            if obj.get("split") != "test":
                continue
            if obj.get("output_language") != "Java":
                continue

            pid = obj.get("problem_id")
            lang = obj.get("input_language")
            src = obj.get("source_code")

            if not pid or not lang or not src:
                continue

            key = (pid, lang)
            if key not in unique:
                unique[key] = {
                    "problem_id": pid,
                    "input_language": lang,
                    "source_code": src.strip(),
                }

    records = list(unique.values())
    print(f"[INFO] Loaded {len(records)} UNIQUE test instances (deduplicated)")

    # Hard safety check
    assert len(records) == 1308, (
        f"ERROR: Expected 1308 test instances, got {len(records)}"
    )

    return records

# =====================================================
# MODEL-AWARE JAVA EXTRACTION
# =====================================================

def extract_first_java_class(text, model_name):
    """
    Model-aware Java extraction.

    CodeT5 models:
        - Cleaner outputs
        - Conservative extraction

    StarCoder / Qwen / DeepSeek:
        - Markdown, chatter, multiple classes
        - Aggressive first-class extraction
    """
    if not text:
        return None

    # Remove markdown fences
    text = re.sub(r"```java|```", "", text, flags=re.IGNORECASE).strip()
    lines = text.splitlines()

    # -----------------------------
    # CodeT5-style models
    # -----------------------------
    if "CodeT5" in model_name:
        collected = []
        brace = 0
        started = False

        for line in lines:
            if "class " in line:
                started = True
            if started:
                collected.append(line)
                brace += line.count("{")
                brace -= line.count("}")
                if brace == 0 and collected:
                    break

        code = "\n".join(collected).strip()
        if "class " in code and code.count("{") == code.count("}"):
            return code
        return None

    # -----------------------------
    # LLM-style models
    # -----------------------------
    start = None
    brace = 0
    collected = []

    for i, line in enumerate(lines):
        if "class " in line:
            start = i
            break

    if start is None:
        return None

    for line in lines[start:]:
        collected.append(line)
        brace += line.count("{")
        brace -= line.count("}")
        if brace == 0:
            break

    code = "\n".join(collected).strip()
    if "class " not in code or code.count("{") != code.count("}"):
        return None

    return code


def prepend_imports(java_code):
    if java_code is None:
        return None
    return DEFAULT_IMPORTS + java_code

# =====================================================
# MODEL LOADER
# =====================================================

def load_model(model_path):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if config.is_encoder_decoder:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model_type = "seq2seq"
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        )
        model_type = "causal"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    model.to(DEVICE).eval()
    return model, tokenizer, model_type

# =====================================================
# GENERATION
# =====================================================

@torch.no_grad()
def generate(model, tokenizer, model_type, enc, sample=False):
    if model_type == "seq2seq":
        out = model.generate(
            **enc,
            max_length=MAX_GEN_LEN,
            do_sample=sample,
            temperature=0.7 if sample else None,
            top_p=0.95 if sample else None
        )
        return tokenizer.decode(out[0], skip_special_tokens=True)

    out = model.generate(
        input_ids=enc["input_ids"],
        attention_mask=enc.get("attention_mask"),
        max_new_tokens=MAX_GEN_LEN,
        do_sample=sample,
        temperature=0.7 if sample else None,
        top_p=0.95 if sample else None,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(
        out[0][enc["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    test_samples = load_test_samples_dedup()

    for model_path in MODEL_PATHS:

        model_name = os.path.basename(model_path.rstrip("/"))
        output_path = os.path.join(
            OUTPUT_DIR, f"{model_name}_predictions.jsonl"
        )

        print(f"\n===== RUNNING MODEL: {model_name} =====")

        model, tokenizer, model_type = load_model(model_path)

        with open(output_path, "w", encoding="utf-8") as fout:
            for sample in test_samples:

                prompt = PROMPT_TEMPLATE.format(
                    input_language=sample["input_language"],
                    source_code=sample["source_code"]
                )

                enc = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(DEVICE)

                record = {
                    "problem_id": sample["problem_id"],
                    "input_language": sample["input_language"]
                }

                # -------- GREEDY --------
                raw = generate(model, tokenizer, model_type, enc, sample=False)
                greedy = prepend_imports(
                    extract_first_java_class(raw, model_name)
                )
                record["greedy"] = greedy
                greedy_status = "OK" if greedy else "NULL"

                # -------- STOCHASTIC --------
                cand_status = []
                for i in range(1, NUM_CANDIDATES + 1):
                    raw = generate(model, tokenizer, model_type, enc, sample=True)
                    cand = prepend_imports(
                        extract_first_java_class(raw, model_name)
                    )
                    record[f"candidate_{i}"] = cand
                    cand_status.append("OK" if cand else "NULL")

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                os.fsync(fout.fileno())

                print(
                    f"[{model_name}] problem_id={sample['problem_id']} | "
                    f"greedy={greedy_status} | cands={cand_status}"
                )

        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()

        print(f"[{model_name}] COMPLETED & CUDA CLEARED")

    print("\nALL MODELS COMPLETED SUCCESSFULLY.")
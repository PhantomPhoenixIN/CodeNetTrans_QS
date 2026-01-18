#!/usr/bin/env python3
"""
Inference script:
Run Score-1 / Score-2 / Score-3 CodeT5 models
on the ENTIRE test set (deduplicated by problem_id + input_language)
and store Java translations.

- Teacher-agnostic
- Flushes output every 10 instances
- Safe for HPC execution
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from accelerate import Accelerator

# =====================================================
# ================= USER CONFIG =======================
# =====================================================

DATASET_PATH = (
    "/home/swaminathanj/TransRectify_New/content/datasets/"
    "codenet_combined_translator_dataset_astfixed.jsonl"
)

MODEL_ROOT = "/home/swaminathanj/TransRectify_New/models/CodeT5_Translation_full_finetune"

OUTPUT_DIR = (
    "/home/swaminathanj/TransRectify_New/content/datasets/"
    "complete_test_set_predictions_FFT"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATHS = {
    # 1: f"{MODEL_ROOT}/score_1/best",
    # 2: f"{MODEL_ROOT}/score_2/best",
    3: f"{MODEL_ROOT}/score_3/best",
}

PROMPT_MAX_LEN = 512
MAX_NEW_TOKENS = 512
FLUSH_EVERY = 10
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
# ========== LOAD DEDUPLICATED TEST SET ================
# =====================================================

def load_full_test_set(path):
    """
    Load ENTIRE test set and deduplicate by
    (problem_id, input_language).
    """
    unique = {}

    with open(path, "r", encoding="utf-8") as f:
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

            # Keep only one copy per problem-language pair
            if key not in unique:
                unique[key] = {
                    "problem_id": pid,
                    "input_language": lang,
                    "source_code": src.strip(),
                }

    records = list(unique.values())
    print(
        f"Loaded {len(records)} UNIQUE test instances "
        "(deduplicated across teachers & scores)"
    )
    return records

# =====================================================
# ================= INFERENCE LOOP ====================
# =====================================================

def run_inference(score, model_path, test_records):
    print(f"\n=== Running inference | SCORE = {score} ===")
    print(f"Model path : {model_path}")

    accelerator = Accelerator(
        mixed_precision="fp16" if USE_FP16 else None
    )
    device = accelerator.device

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    model = accelerator.prepare(model)
    model.eval()

    out_path = os.path.join(
        OUTPUT_DIR,
        f"test_score{score}_model_translation.jsonl",
    )

    fout = open(out_path, "w", encoding="utf-8")

    with torch.no_grad():
        for idx, obj in enumerate(
            tqdm(test_records, desc=f"Score-{score} inference")
        ):
            prompt = PROMPT_TEMPLATE.format(
                input_language=obj["input_language"],
                source_code=obj["source_code"],
            )

            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=PROMPT_MAX_LEN,
            ).to(device)

            gen = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            pred_java = tokenizer.decode(
                gen[0], skip_special_tokens=True
            ).strip()

            out_obj = {
                "problem_id": obj["problem_id"],
                "input_language": obj["input_language"],
                "source_code": obj["source_code"],
                "predicted_java": pred_java,
                "model_score": score,
            }

            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

            # Flush every N instances (HPC safety)
            if (idx + 1) % FLUSH_EVERY == 0:
                fout.flush()

    fout.close()
    print(f"Saved predictions → {out_path}")

# =====================================================
# ========================= MAIN ======================
# =====================================================

def main():
    test_records = load_full_test_set(DATASET_PATH)

    for score, model_path in MODEL_PATHS.items():
        run_inference(score, model_path, test_records)

    print("\nAll inference runs completed successfully.")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Generate model predictions on a test set for evaluation.

Robustness features:
- Incremental append: each prediction is written and flushed immediately, so an
  interrupted run keeps its progress instead of losing everything.
- Resume: prompts already present in the output file are skipped, so re-running
  the same command continues where it left off.
- Retries: transient HTTP/connection errors are retried with backoff; only a
  persistent failure is recorded as an "Error: ..." prediction.

The API URL defaults to $ORIN_API_URL (see configs/deploy.env), then to the
built-in fallback, and can be overridden with --api_url.
"""

import os
import json
import time
import argparse
import requests
from pathlib import Path
from tqdm import tqdm

DEFAULT_API_URL = os.environ.get("ORIN_API_URL", "http://30.245.40.99:8080")
DEFAULT_MODEL = os.environ.get("ORIN_MODEL", "qwen3-4b-q4_k_m.gguf")


def call_model(prompt, api_url, model, max_tokens=512, retries=3, timeout=60):
    """Call the llama.cpp server, retrying transient failures with backoff."""
    last_err = None
    for attempt in range(retries):
        try:
            response = requests.post(
                f"{api_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.1,
                },
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001 - record and retry any request error
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return f"Error: {last_err}"


def load_done_prompts(output_path):
    """Return the set of prompts already successfully predicted (for resume)."""
    done = set()
    if not output_path.exists():
        return done
    with open(output_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Only treat non-error predictions as done so failures get retried.
            if rec.get("prediction") and not rec["prediction"].startswith("Error:"):
                done.add(rec.get("prompt"))
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--api_url", type=str, default=DEFAULT_API_URL)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max_samples", type=int, default=100)
    args = parser.parse_args()

    test_data = [json.loads(l) for l in open(args.test_file)][:args.max_samples]

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_prompts(output_path)
    if done:
        print(f"Resuming: {len(done)} predictions already done, skipping them.")

    written = 0
    # Append mode preserves prior good predictions; we skip their prompts above.
    with open(output_path, "a", encoding="utf-8") as out:
        for item in tqdm(test_data, desc="Generating predictions"):
            prompt = next((m["content"] for m in item["messages"]
                           if m["role"] == "user"), "")
            if prompt in done:
                continue
            pred_text = call_model(prompt, args.api_url, args.model)
            out.write(json.dumps({"prompt": prompt, "prediction": pred_text},
                                 ensure_ascii=False) + "\n")
            out.flush()
            written += 1

    print(f"Wrote {written} new predictions to {output_path}")


if __name__ == "__main__":
    main()

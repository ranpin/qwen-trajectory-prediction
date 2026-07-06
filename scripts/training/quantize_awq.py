#!/usr/bin/env python3
"""Quantize Qwen3-4B trajectory model with AWQ 4-bit."""

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to merged model (e.g., outputs/qwen3-4b-trajectory-lora-merged)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output path for quantized model (default: {model_path}-awq)")
    parser.add_argument("--w_bit", type=int, default=4, help="Quantization bits")
    parser.add_argument("--q_group_size", type=int, default=128, help="Quantization group size")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    output_path = Path(args.output_path) if args.output_path else model_path.parent / f"{model_path.name}-awq"

    if not model_path.exists():
        print(f"Error: Model path {model_path} does not exist!")
        return

    print(f"Quantizing model: {model_path}")
    print(f"Output: {output_path}")
    print(f"Bits: {args.w_bit}, Group size: {args.q_group_size}")

    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError:
        print("AWQ not installed. Install with: pip install autoawq")
        return

    print("Loading model...")
    model = AutoAWQForCausalLM.from_pretrained(str(model_path))
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    print("Quantizing...")
    quant_config = {
        "zero_point": True,
        "q_group_size": args.q_group_size,
        "w_bit": args.w_bit,
    }
    model.quantize(tokenizer, quant_config)

    print(f"Saving quantized model to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    print(f"Quantization complete! Model saved to {output_path}")

    # Calculate size
    total_size = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
    print(f"Model size: {total_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()

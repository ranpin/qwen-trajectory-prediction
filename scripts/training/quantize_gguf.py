#!/usr/bin/env python3
"""Quantize Qwen3-4B trajectory model to GGUF format for llama.cpp."""

import os
import argparse
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to merged model (e.g., outputs/qwen3-4b-trajectory-lora-merged)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output path for GGUF model (default: {model_path}.gguf)")
    parser.add_argument("--quant_type", type=str, default="Q4_K_M",
                        choices=["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q8_0", "F16"],
                        help="Quantization type")
    parser.add_argument("--llama_cpp_path", type=str, default=None,
                        help="Path to llama.cpp directory (will clone if not provided)")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    output_path = Path(args.output_path) if args.output_path else model_path.parent / f"{model_path.name}-{args.quant_type.lower()}.gguf"

    if not model_path.exists():
        print(f"Error: Model path {model_path} does not exist!")
        sys.exit(1)

    print(f"Converting model: {model_path}")
    print(f"Output: {output_path}")
    print(f"Quantization: {args.quant_type}")

    # Locate an existing llama.cpp before cloning: --llama_cpp_path, then
    # $LLAMA_CPP_PATH, then common locations. Only clone/build as a last resort.
    candidates = []
    if args.llama_cpp_path:
        candidates.append(Path(args.llama_cpp_path))
    if os.environ.get("LLAMA_CPP_PATH"):
        candidates.append(Path(os.environ["LLAMA_CPP_PATH"]))
    candidates += [Path.home() / "llama.cpp", Path("/data/tmp/chenrunbin/llama.cpp")]

    llama_cpp_path = next(
        (p for p in candidates if (p / "convert_hf_to_gguf.py").exists()), None)
    if llama_cpp_path is None:
        llama_cpp_path = candidates[0]
        print(f"Cloning llama.cpp to {llama_cpp_path}...")
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/ggerganov/llama.cpp.git",
                        str(llama_cpp_path)], check=True)
        subprocess.run(["cmake", "-B", str(llama_cpp_path / "build"),
                        "-S", str(llama_cpp_path)], check=True)
        subprocess.run(["cmake", "--build", str(llama_cpp_path / "build"),
                        "--target", "llama-quantize", "-j"], check=True)
    print(f"Using llama.cpp at {llama_cpp_path}")

    # Step 1: Convert to FP16 GGUF
    f16_path = output_path.parent / f"{model_path.name}-f16.gguf"
    print(f"Step 1: Converting to FP16 GGUF...")
    convert_script = llama_cpp_path / "convert_hf_to_gguf.py"

    # Use the current interpreter (the venv) so the convert script's deps
    # (gguf/torch/safetensors) resolve — system python3 usually lacks them.
    subprocess.run([
        sys.executable, str(convert_script),
        str(model_path),
        "--outfile", str(f16_path),
        "--outtype", "f16"
    ], check=True)

    print(f"FP16 GGUF saved to {f16_path}")

    # Step 2: Quantize
    if args.quant_type != "F16":
        print(f"Step 2: Quantizing to {args.quant_type}...")
        # CMake builds put the binary in build/bin; old Makefile put it at root.
        quantize_bin = next(
            (b for b in [llama_cpp_path / "build" / "bin" / "llama-quantize",
                         llama_cpp_path / "llama-quantize"] if b.exists()), None)
        if quantize_bin is None:
            print("Error: llama-quantize binary not found in llama.cpp")
            sys.exit(1)

        subprocess.run([
            str(quantize_bin),
            str(f16_path),
            str(output_path),
            args.quant_type
        ], check=True)

        print(f"Quantized model saved to {output_path}")

        # Remove FP16 intermediate
        f16_path.unlink()
    else:
        # Just rename FP16 to output
        f16_path.rename(output_path)

    # Calculate size
    total_size = output_path.stat().st_size
    print(f"Final model size: {total_size / 1024 / 1024:.1f} MB")
    print(f"GGUF quantization complete!")


if __name__ == "__main__":
    main()

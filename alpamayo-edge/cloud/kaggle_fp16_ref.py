# Kaggle script kernel — Cosmos-Reason2-8B FP16 accuracy reference.
#
# Purpose: produce the FP16 baseline the Orin INT4/INT8 deployment lacks, so we can
# quantify quantization "掉点". Two metrics, both computed here:
#   (1) greedy FP16 reference generations on a fixed prompt set (same prompts the
#       Orin engines ran with top_k=1 greedy), then token-level agreement +
#       first-divergence position of the INT4/INT8 outputs vs this FP16 reference.
#   (2) teacher-forced perplexity of each deployed output (int4 / int8) AND of the
#       FP16 reference's own output, all scored under the SAME FP16 model — i.e.
#       "how surprised is the original model by the quantized output". PPL ratio
#       vs the fp16-own output is a clean degradation number.
#
# Runs text-only (no images) — the LLM backbone is what gets quantized; the visual
# tower stays fp16 in both int4/int8 deployments, so text is the fair quant axis.
#
# PREREQ: GPU on (T4x2 ideal; single-GPU falls back to CPU offload via device_map).
#         HF_TOKEN available (env — injected at push time for API runs).
import os, json, subprocess, sys

# ---- HF auth (env; a token line may be prepended by the push wrapper) ----
assert os.environ.get("HF_TOKEN"), "HF_TOKEN not set"
os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def sh(c):
    print("+", c, flush=True); subprocess.run(c, shell=True, check=True)

# Pin a transformers new enough for Qwen2.5-VL / Cosmos-Reason2 remote code.
sh("pip -q install -U 'transformers>=4.51' accelerate safetensors 2>&1 | tail -3 || true")

import torch
from transformers import AutoTokenizer, AutoProcessor

# Run on GPU. IMPORTANT: push with machine_shape="NvidiaTeslaT4" so Kaggle
# provisions T4x2 (sm_75, 32GB across 2 cards). Without it the API default is a
# single Tesla P100 (sm_60), which the preinstalled PyTorch (sm_70..sm_120)
# cannot run -> "no kernel image". device_map="auto" spreads the 16GB fp16 model
# across the two T4s with no CPU offload.
DTYPE = torch.float16

MODEL = "nvidia/Cosmos-Reason2-8B"
MAXNEW = 160

# Same 12 prompts the Orin engines ran (greedy, top_k=1).
PROMPTS = [
    "In one paragraph, explain why braking distance increases on wet roads.",
    "A pedestrian steps into the crosswalk ahead while your vehicle is moving at 40 km/h. List the sequence of driving actions you should take.",
    "Explain what hydroplaning is and how a driver can reduce the risk of it.",
    "Why does stopping distance grow roughly with the square of speed? Explain the physics.",
    "You are driving at night on an unlit rural road. List the main hazards and how to mitigate each.",
    "Describe the correct procedure if your vehicle begins to skid on black ice.",
    "Explain the two-second following-distance rule and why it should increase in poor conditions.",
    "A cyclist is riding in your lane on a narrow road. Explain how and when to overtake them safely.",
    "What should a driver do immediately if a front tire blows out at highway speed?",
    "Explain how anti-lock braking systems (ABS) help a driver maintain control during hard braking.",
    "In two or three sentences, state Newton's second law and give an everyday example.",
    "Briefly explain the difference between weather and climate.",
]

# Deployed Orin greedy outputs, indexed by prompt position. Injected after the
# Orin run; if empty the kernel still produces the FP16 reference + self-PPL.
# Orin greedy outputs injected at push time (json keyed by prompt idx -> {int4,int8}); {} = ref-only.
CANDIDATES = {}

# ---- load tokenizer/processor + model (robust to class + gpu-count) ----
try:
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    tok = getattr(proc, "tokenizer", None) or proc
except Exception as e:
    print("[warn] AutoProcessor failed, using AutoTokenizer:", e, flush=True)
    proc = None
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

def load_model():
    last = None
    for cls_name in ("AutoModelForImageTextToText", "AutoModelForCausalLM", "AutoModelForVision2Seq"):
        try:
            import transformers as T
            cls = getattr(T, cls_name)
            m = cls.from_pretrained(
                MODEL, dtype=DTYPE, device_map="auto",
                trust_remote_code=True, low_cpu_mem_usage=True,
            )
            print(f"[load] {cls_name} OK", flush=True)
            return m
        except Exception as e:
            print(f"[load] {cls_name} failed: {repr(e)[:200]}", flush=True)
            last = e
    raise last

subprocess.run("nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader", shell=True)
model = load_model()
model.eval()

def fmt(prompt):
    msgs = [{"role": "user", "content": prompt}]
    tmpl = (proc or tok).apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tmpl

@torch.inference_mode()
def gen_greedy(prompt):
    text = fmt(prompt)
    ids = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=MAXNEW, do_sample=False, num_beams=1,
                          pad_token_id=tok.pad_token_id or tok.eos_token_id)
    gen = out[0][ids["input_ids"].shape[1]:]
    return text, tok.decode(gen, skip_special_tokens=True)

@torch.inference_mode()
def nll_of(prompt_text, answer_text):
    """teacher-forced mean NLL (nats/token) of answer given the formatted prompt."""
    if not answer_text:
        return None
    p_ids = tok(prompt_text, return_tensors="pt").input_ids
    full = tok(prompt_text + answer_text, return_tensors="pt").input_ids.to(model.device)
    n_prompt = p_ids.shape[1]
    labels = full.clone()
    labels[:, :n_prompt] = -100
    loss = model(input_ids=full, labels=labels).loss
    return float(loss)  # mean NLL over answer tokens

def tok_agree(ref_text, cand_text):
    if not cand_text:
        return None
    a = tok(ref_text, add_special_tokens=False).input_ids
    b = tok(cand_text, add_special_tokens=False).input_ids
    lcp = 0
    for x, y in zip(a, b):
        if x == y: lcp += 1
        else: break
    return {"ref_len": len(a), "cand_len": len(b), "common_prefix": lcp,
            "prefix_frac": (lcp / max(1, len(a)))}

results = []
for i, p in enumerate(PROMPTS):
    ptext, fp16_out = gen_greedy(p)
    if i == 0:
        print("=== formatted prompt[0] (verify matches Orin) ===", flush=True)
        print(repr(ptext), flush=True)
    cand = CANDIDATES.get(str(i), CANDIDATES.get(i, {})) if isinstance(CANDIDATES, dict) else {}
    int4_out = cand.get("int4"); int8_out = cand.get("int8")
    rec = {
        "idx": i, "prompt": p, "fp16_output": fp16_out,
        "ppl_fp16_own": None, "ppl_int4": None, "ppl_int8": None,
        "agree_int4": None, "agree_int8": None,
    }
    import math
    def ppl(x): return None if x is None else math.exp(x)
    rec["ppl_fp16_own"] = ppl(nll_of(ptext, fp16_out))
    if int4_out is not None:
        rec["ppl_int4"] = ppl(nll_of(ptext, int4_out)); rec["agree_int4"] = tok_agree(fp16_out, int4_out)
    if int8_out is not None:
        rec["ppl_int8"] = ppl(nll_of(ptext, int8_out)); rec["agree_int8"] = tok_agree(fp16_out, int8_out)
    results.append(rec)
    print(f"[{i}] ppl fp16={rec['ppl_fp16_own']} int4={rec['ppl_int4']} int8={rec['ppl_int8']}", flush=True)

out = {"model": MODEL, "max_new_tokens": MAXNEW, "n_prompts": len(PROMPTS), "results": results}
with open("/kaggle/working/acc_ref.json", "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("WROTE /kaggle/working/acc_ref.json", flush=True)

# aggregate quick view
def mean(xs): xs=[x for x in xs if x is not None]; return sum(xs)/len(xs) if xs else None
print("MEAN ppl fp16_own=", mean([r["ppl_fp16_own"] for r in results]),
      "int4=", mean([r["ppl_int4"] for r in results]),
      "int8=", mean([r["ppl_int8"] for r in results]), flush=True)
print("DONE", flush=True)

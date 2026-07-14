# Kaggle cell — Cosmos-Reason2-8B INT4/INT8 ONLY (drops Alpamayo-10B which OOMs a 15GB T4).
# Prereq (already set on the kernel): Accelerator=GPU T4 x2, Internet=On, Secret HF_TOKEN.
# Stop the stuck run first, paste this as the notebook body, Save & Run All.
import os, subprocess
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
def sh(c): print("+", c, flush=True); return subprocess.run(c, shell=True, check=True)
sh("git clone --depth 1 https://github.com/NVIDIA/TensorRT-Edge-LLM.git /kaggle/working/TRTEdge")
sh("cd /kaggle/working/TRTEdge && pip -q install '.[tools]'")
os.environ["PYTHONPATH"] = "/kaggle/working/TRTEdge"
os.chdir("/kaggle/working")
subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True)
for QF in ["int4_awq", "int8_sq"]:
    try:
        sh(f"tensorrt-edgellm-quantize llm --model_dir nvidia/Cosmos-Reason2-8B --output_dir Cosmos-8B-{QF} --quantization {QF}")
        sh(f"tensorrt-edgellm-export Cosmos-8B-{QF} Cosmos-8B-{QF}/onnx")
    except Exception as e:
        print(f"COSMOS_{QF}_FAILED:", e, flush=True)
paths = [p for p in ["Cosmos-8B-int4_awq/onnx", "Cosmos-8B-int8_sq/onnx"] if os.path.isdir(p)]
if paths:
    sh("tar -czf /kaggle/working/edge_artifacts.tgz " + " ".join(paths))
    print("PACKED_OK", paths, flush=True)
else:
    print("NO_ONNX_PRODUCED", flush=True)
subprocess.run("rm -rf /kaggle/working/TRTEdge /kaggle/working/Cosmos-8B-int4_awq /kaggle/working/Cosmos-8B-int8_sq", shell=True)
print("DONE", flush=True)

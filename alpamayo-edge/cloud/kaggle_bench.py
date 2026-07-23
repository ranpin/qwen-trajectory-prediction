# Kaggle kernel — Cosmos-Reason1-Benchmark accuracy (FP16 reference) + frame export.
# Runs the official MC benchmark (robovqa+robofail) with the FP16 model, computes
# multiple-choice accuracy, AND exports the sampled frames + questions + answers so
# the Orin INT4/INT8 engines can be scored on the IDENTICAL inputs (fair drop-off).
# Video -> K uniformly-sampled frames fed as multi-image (Orin runtime is image-only;
# verified multi-image works). Same frame protocol on FP16 and Orin => drop-off valid.
import os, json, glob, subprocess, io, re, tarfile
assert os.environ.get("HF_TOKEN"); os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"

def sh(c): print("+",c,flush=True); subprocess.run(c,shell=True,check=True)
sh("pip -q install -U 'transformers>=4.51' accelerate 'huggingface_hub' qwen-vl-utils opencv-python-headless pillow 2>&1 | tail -2")

import torch, cv2, numpy as np
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

MODEL="nvidia/Cosmos-Reason2-8B"; REPO="nvidia/Cosmos-Reason1-Benchmark"
SUBSETS=["robovqa"]; K=6; MAXNEW=12; MAXSIDE=448  # 每帧≤448px→~256 ViT patches;6帧≈1536<Orin ViT 引擎上限4096
OUT="/kaggle/working"; FR=f"{OUT}/frames"; os.makedirs(FR,exist_ok=True)

print("[dl] snapshot benchmark repo...",flush=True)
root=snapshot_download(REPO, repo_type="dataset",
      allow_patterns=[f"{s}/*" for s in SUBSETS])
print("[dl] root=",root,flush=True)
for s in SUBSETS:
    print(s, sorted(os.listdir(os.path.join(root,s)))[:8], flush=True)

def load_rows(subset):
    d=os.path.join(root,subset)
    # extract clips.tar.gz -> d/clips/*.mp4 (once)
    tgz=os.path.join(d,"clips.tar.gz")
    if os.path.exists(tgz) and not os.path.isdir(os.path.join(d,"clips")):
        with tarfile.open(tgz) as t: t.extractall(d)
    # QA annotations live in a *_qa_pairs.json (a list of records)
    jf=glob.glob(os.path.join(d,"*qa_pairs*.json"))[0]
    data=json.load(open(jf))
    recs = data if isinstance(data,list) else (data.get("data") or list(data.values()))
    # flatten: each record has a video + one-or-more qa_pairs
    rows=[]
    for r in recs:
        vid=r.get("video") or r.get("video_path") or r.get("clip")
        qp=r.get("qa_pairs")
        for qa in (qp if isinstance(qp,list) else [qp]):
            if qa: rows.append({"video":vid,"qa_pairs":qa})
    print(f"[{subset}] json={os.path.basename(jf)} records={len(recs)} qa_rows={len(rows)}",flush=True)
    return d, rows

def _rs(im):
    im=im.convert("RGB"); im.thumbnail((MAXSIDE,MAXSIDE)); return im  # 缩放以适配 Orin ViT patch 上限

def sample_frames(path, k):
    cap=cv2.VideoCapture(path); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if n<=0:
        frames=[];
        while True:
            ok,f=cap.read()
            if not ok: break
            frames.append(f)
        cap.release()
        if not frames: return []
        idx=np.linspace(0,len(frames)-1,min(k,len(frames))).astype(int)
        return [_rs(Image.fromarray(cv2.cvtColor(frames[i],cv2.COLOR_BGR2RGB))) for i in idx]
    idx=np.linspace(0,n-1,min(k,n)).astype(int); out=[]
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(i)); ok,f=cap.read()
        if ok: out.append(_rs(Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))))
    cap.release(); return out

proc=AutoProcessor.from_pretrained(MODEL,trust_remote_code=True)
tok=getattr(proc,"tokenizer",None) or proc
model=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.float16,device_map="auto",trust_remote_code=True,low_cpu_mem_usage=True).eval()
subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",shell=True)

def build_prompt(q,opts):
    lines=[f"{L}. {opts[L]}" for L in sorted(opts)]
    return (f"You are shown {K} frames sampled in order from a video.\n{q}\n"
            f"Options:\n"+"\n".join(lines)+"\nAnswer with ONLY the single letter of the correct option.")

def parse_letter(t,opts):
    m=re.search(r"[A-D]", (t or "").strip().upper())
    return m.group(0) if m and m.group(0) in opts else None

@torch.inference_mode()
def infer(frames,q,opts):
    prompt=build_prompt(q,opts)
    content=[{"type":"image","image":im} for im in frames]+[{"type":"text","text":prompt}]
    msgs=[{"role":"user","content":content}]
    text=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    imgs,vids=process_vision_info(msgs)
    inp=proc(text=[text],images=imgs,videos=vids,padding=True,return_tensors="pt").to(model.device)
    g=model.generate(**inp,max_new_tokens=MAXNEW,do_sample=False)
    gen=g[0][inp["input_ids"].shape[1]:]
    return tok.decode(gen,skip_special_tokens=True)

manifest=[]; per={}
for subset in SUBSETS:
    base,rows=load_rows(subset); ok=0; tot=0
    for i,row in enumerate(rows):
        try:
            qa=row["qa_pairs"]; qa=qa if isinstance(qa,dict) else json.loads(qa)
            opts=dict(qa["index2ans"]); opts={k:v for k,v in opts.items() if v is not None and str(v)!=""}
            gt=qa["answer"]; q=qa["question"]
            vp=row["video"];
            cand=[os.path.join(base,vp),os.path.join(base,os.path.basename(vp)),
                  os.path.join(root,vp)]
            vpath=next((c for c in cand if os.path.exists(c)),None)
            if not vpath:
                print("  miss video",vp,flush=True); continue
            frames=sample_frames(vpath,K)
            if not frames: print("  no frames",vp,flush=True); continue
            out=infer(frames,q,opts); pred=parse_letter(out,opts)
            tot+=1; ok+= int(pred==gt)
            # save frames for Orin
            fps=[]
            for j,im in enumerate(frames):
                fn=f"{subset}_{i}_{j}.jpg"; im.convert("RGB").save(f"{FR}/{fn}",quality=88); fps.append(fn)
            manifest.append({"subset":subset,"idx":int(i),"question":q,"options":opts,
                             "answer":gt,"fp16_pred":pred,"fp16_raw":out[:80],"frames":fps})
        except Exception as e:
            print("  err",i,repr(e)[:120],flush=True)
        if tot and tot%25==0: print(f"  [{subset}] {ok}/{tot}={ok/tot:.3f}",flush=True)
    per[subset]={"correct":ok,"total":tot,"acc":ok/tot if tot else None}
    print(f"[{subset}] FP16 acc = {ok}/{tot} = {per[subset]['acc']}",flush=True)

allok=sum(p["correct"] for p in per.values()); alltot=sum(p["total"] for p in per.values())
summary={"model":MODEL,"K":K,"subsets":per,"overall_fp16_acc":allok/alltot if alltot else None,"n":alltot}
json.dump({"summary":summary,"manifest":manifest},open(f"{OUT}/bench_fp16.json","w"),indent=2,ensure_ascii=False)
print("OVERALL FP16 ACC:",summary["overall_fp16_acc"],"n=",alltot,flush=True)

# pack frames + manifest for Orin
with tarfile.open(f"{OUT}/bench_frames.tgz","w:gz") as t:
    t.add(FR,arcname="frames"); t.add(f"{OUT}/bench_fp16.json",arcname="bench_fp16.json")
import shutil; shutil.rmtree(FR,ignore_errors=True)
print("PACKED bench_frames.tgz ; DONE",flush=True)

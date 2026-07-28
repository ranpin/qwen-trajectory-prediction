#!/usr/bin/env python3
"""从 Kaggle kernel 拉大产物（>1 GB）—— 绕开两个已验证的坑，见 docs/PROBLEMS.md E1 续。

坑 1：`kaggle kernels output` 把整个响应读进内存，6.46 GiB 产出 0 字节且退出码 0（静默失败）。
坑 2：`curl -C -` 配"每次重取的短时效签名 URL"会静默损坏文件（长到 8.29 GB，zlib invalid block type）。

做法：Range: bytes=0-0 探测（kaggleusercontent 不支持 HEAD，只能从 Content-Range 读总长）
      -> 按显式字节区间分段拉，每段单独重取新签名 URL -> 顺序追加 -> 服务端忽略 Range 就中止。
用法：改下面的 kernelSlug / OUT 后 `.venv/bin/python cloud/pull_kernel_output.py`，
      结束后**务必**用退出码校验：`tar tzf <file> >/dev/null && echo OK`（别用管道，会吞退出码）。
"""
import json, os, sys, time

import requests

C=json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))
AUTH=(C["username"], C["key"])
OUT="/Users/ranpin/projects/qwen-trajectory-prediction/alpamayo-edge/outputs/edge_artifacts_lmhead.tgz"
CHUNK=400<<20   # 400 MB/段，每段重取一次签名 URL，规避短时效
def fresh_url():
    r=requests.get("https://www.kaggle.com/api/v1/kernels/output",
        params={"userName":"mamihlapinatapai","kernelSlug":"alpamayo-edge-lmhead-2b8b"},
        auth=AUTH, timeout=60)
    for f in r.json().get("files",[]):
        if f.get("fileName","").endswith(".tgz"): return f["url"]
    raise RuntimeError("no tgz url")
# kaggleusercontent 不支持 HEAD，用 Range: bytes=0-0 从 Content-Range 读总长
probe=requests.get(fresh_url(), headers={"Range":"bytes=0-0"}, stream=True, timeout=60)
cr=probe.headers.get("Content-Range",""); probe.close()
if probe.status_code!=206 or "/" not in cr: sys.exit(f"Range 不可用: {probe.status_code} {cr!r}")
total=int(cr.rsplit("/",1)[1])
print(f"total={total:,} ({total/2**30:.2f} GiB)  Range OK (206)", flush=True)
have=os.path.getsize(OUT) if os.path.exists(OUT) else 0
mode="ab" if have else "wb"
print(f"已有 {have/1e6:.0f} MB，从此处继续", flush=True)
with open(OUT, mode) as f:
    while have < total:
        end=min(have+CHUNK, total)-1
        for att in range(6):
            try:
                r=requests.get(fresh_url(), headers={"Range":f"bytes={have}-{end}"},
                               stream=True, timeout=(30,300))
                if r.status_code not in (206,200): raise RuntimeError(f"HTTP {r.status_code}")
                if r.status_code==200 and have>0: raise RuntimeError("服务端忽略 Range，不能安全续传")
                n=0
                for b in r.iter_content(1<<20):
                    f.write(b); n+=len(b)
                f.flush()
                have+=n
                print(f"  {have/1e9:.2f}/{total/1e9:.2f} GB ({100*have/total:.1f}%)", flush=True)
                break
            except Exception as e:
                print(f"  段 {have}-{end} 第 {att+1} 次失败: {type(e).__name__}: {e}", flush=True)
                time.sleep(5)
        else:
            sys.exit("该段反复失败，停止")
print(f"DOWNLOAD_COMPLETE {os.path.getsize(OUT)} bytes", flush=True)

"""Day-0 Modal sanity check: auth + GPU allocation + volume mount.
No heavy image, no torch download — nvidia-smi is injected by Modal on GPU.

  modal run scripts/modal_hello.py
"""

import os
import subprocess

import modal

app = modal.App("geoql-hello")
image = modal.Image.debian_slim()
vol_data = modal.Volume.from_name("geoql-data", create_if_missing=True)
vol_ckpt = modal.Volume.from_name("geoql-ckpts", create_if_missing=True)


@app.function(gpu="L40S", image=image,
              volumes={"/data": vol_data, "/ckpts": vol_ckpt})
def check():
    print("=== nvidia-smi ===")
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader"])
    print("=== volume mounts ===")
    print("/data  ->", sorted(os.listdir("/data"))[:10] or "EMPTY (upload pending)")
    print("/ckpts ->", sorted(os.listdir("/ckpts"))[:10] or "EMPTY (expected on Day 0)")
    return "ok"


@app.local_entrypoint()
def main():
    print("result:", check.remote())

"""Task 7 - deploy the model API (api/) to Vercel through the REST API.

Uploads only the files the function needs (code, pinned requirements, TFLite
model + metadata), creates a production deployment and waits until it is ready.
Credentials come from .env (VERCEL_TOKEN); API_KEY is stored as an encrypted
Vercel environment variable, never in the uploaded files.

  python src/deploy_vercel.py
"""
import hashlib
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT = os.getenv("VERCEL_PROJECT", "23-l-0871-a2-ops")
FILES = ["index.py", "preprocess.py", "requirements.txt", "vercel.json",
         "model/champion.tflite", "model/champion.json"]
API = "https://api.vercel.com"


def main():
    load_dotenv()
    h = {"Authorization": f"Bearer {os.environ['VERCEL_TOKEN']}"}
    project = requests.get(f"{API}/v9/projects/{PROJECT}", headers=h, timeout=30).json()
    q = {"teamId": project["accountId"]}
    root = project.get("rootDirectory") or ""

    refs = []
    for rel in FILES:
        data = (Path("api") / rel).read_bytes()
        sha = hashlib.sha1(data).hexdigest()
        r = requests.post(f"{API}/v2/files", params=q, data=data, timeout=300,
                          headers={**h, "x-vercel-digest": sha, "Content-Type": "application/octet-stream"})
        r.raise_for_status()
        refs.append({"file": f"{root}/{rel}" if root else rel, "sha": sha, "size": len(data)})
        print(f"uploaded {rel} ({len(data) / 1e6:.2f} MB)")

    r = requests.post(f"{API}/v13/deployments", params=q, headers=h, timeout=60, json={
        "name": PROJECT, "project": project["id"], "target": "production", "files": refs,
        "meta": {"source": "src/deploy_vercel.py"}})
    r.raise_for_status()
    dep = r.json()
    print(f"deployment {dep['id']} created: https://{dep['url']}")
    while True:
        dep = requests.get(f"{API}/v13/deployments/{dep['id']}", params=q, headers=h, timeout=30).json()
        state = dep.get("readyState")
        print(f"  state: {state}")
        if state in ("READY", "ERROR", "CANCELED"):
            break
        time.sleep(10)
    if state != "READY":
        raise SystemExit("deployment failed - inspect the build logs in the Vercel dashboard")
    print(f"production URL: https://{PROJECT}.vercel.app")


if __name__ == "__main__":
    main()

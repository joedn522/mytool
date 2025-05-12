#!/usr/bin/env bash
set -e

PATCH_URL="https://raw.githubusercontent.com/joedn522/mytool/main/mov_support_and_5s_extraction.patch"

python - <<'PY'
import pathlib, subprocess, vbench, urllib.request, tempfile, os, sys
url = "https://raw.githubusercontent.com/joedn522/mytool/main/mov_support_and_5s_extraction.patch"
site_dir = pathlib.Path(vbench.__file__).parent.parent          # …/site‑packages
print(f"Applying patch to {site_dir} ...")

patch_data = urllib.request.urlopen(url, timeout=30).read()
with tempfile.NamedTemporaryFile(delete=False) as tmp:
    tmp.write(patch_data); tmp_path = tmp.name

subprocess.check_call(["patch", "-p1", "-d", str(site_dir), "-i", tmp_path])
print("✓ Patch applied OK")
PY

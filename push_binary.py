#!/usr/bin/env python3
"""
Push the binary files (model weights + sample images) to your RestorNet-S
GitHub repo using the GitHub Contents API.

Run this YOURSELF, from your own computer, inside the unzipped
`restornet-s` folder. Your token is entered securely (not shown on
screen, not saved anywhere, never sent to Claude) and used only for
these API calls, made directly from your machine to GitHub.

Usage:
    cd restornet-s          # the unzipped project folder
    python push_binaries.py

Requires: pip install requests
"""
import base64
import getpass
import os
import sys

try:
    import requests
except ImportError:
    print("Missing dependency. Run:  pip install requests")
    sys.exit(1)

REPO = "Riderneal/RestorNet-S"
BRANCH = "main"
API = f"https://api.github.com/repos/{REPO}/contents"

# (local relative path, path in the repo)
FILES = [
    ("weights/restornet_s_final.pth", "weights/restornet_s_final.pth"),
    ("sample_outputs/comparison_grid.png", "sample_outputs/comparison_grid.png"),
]
for i in range(6):
    n = f"sample_{i:02d}.png"
    FILES.append((f"sample_outputs/degraded/{n}", f"sample_outputs/degraded/{n}"))
    FILES.append((f"sample_outputs/ground_truth/{n}", f"sample_outputs/ground_truth/{n}"))
    rn = f"sample_{i:02d}_restored.png"
    FILES.append((f"sample_outputs/restored/{rn}", f"sample_outputs/restored/{rn}"))


def push_file(session: requests.Session, local_path: str, repo_path: str) -> None:
    if not os.path.exists(local_path):
        print(f"  SKIP (not found): {local_path}")
        return

    with open(local_path, "rb") as f:
        content_b64 = base64.b64encode(f.read()).decode("ascii")

    url = f"{API}/{repo_path}"

    # Check if the file already exists (need its sha to update it)
    sha = None
    r = session.get(url, params={"ref": BRANCH})
    if r.status_code == 200:
        sha = r.json().get("sha")

    payload = {
        "message": f"Add {repo_path}",
        "content": content_b64,
        "branch": BRANCH,
    }
    if sha:
        payload["sha"] = sha

    r = session.put(url, json=payload)
    if r.status_code in (200, 201):
        print(f"  OK: {repo_path}")
    else:
        print(f"  FAILED ({r.status_code}): {repo_path} -> {r.text[:200]}")


def main():
    token = getpass.getpass("Paste your GitHub token (input hidden, not stored): ").strip()
    if not token:
        print("No token entered, aborting.")
        sys.exit(1)

    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    })

    print(f"\nPushing {len(FILES)} files to {REPO} ...\n")
    for local_path, repo_path in FILES:
        push_file(session, local_path, repo_path)

    print("\nDone. You can now revoke the token at "
          "https://github.com/settings/tokens?type=beta")


if __name__ == "__main__":
    main()

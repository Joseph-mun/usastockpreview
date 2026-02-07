# -*- coding: utf-8 -*-
"""GitHub Release storage for model files and SMA cache."""

import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from src.config import MODEL_DIR, SMA_CACHE_DIR


RELEASE_TAG = "model-latest"
RELEASE_TITLE = "Model & SMA Cache (auto-updated)"


def _run_gh(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a `gh` CLI command."""
    result = subprocess.run(
        ["gh"] + args,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"gh command failed: {result.stderr.strip()}")
    return result


def _ensure_release():
    """Create the release tag if it doesn't exist."""
    result = _run_gh(["release", "view", RELEASE_TAG], check=False)
    if result.returncode != 0:
        _run_gh([
            "release", "create", RELEASE_TAG,
            "--title", RELEASE_TITLE,
            "--notes", "Auto-managed release for model artifacts.",
        ])


def upload_artifacts(verbose: bool = True):
    """
    Upload model files (.joblib, .json) and SMA cache (.zip) to GitHub Release.
    Overwrites existing assets with the same name.
    """
    _ensure_release()

    files_to_upload = []

    # Model files
    for ext in ("*.joblib", "*_meta.json"):
        files_to_upload.extend(glob.glob(str(MODEL_DIR / ext)))

    # SMA cache (latest only)
    cache_zips = sorted(glob.glob(str(SMA_CACHE_DIR / "sma_cache_*.zip")))
    if cache_zips:
        files_to_upload.append(cache_zips[-1])

    if not files_to_upload:
        if verbose:
            print("WARNING: No artifacts found to upload.")
        return

    if verbose:
        print(f"Uploading {len(files_to_upload)} files to release '{RELEASE_TAG}':")
        for f in files_to_upload:
            print(f"  {Path(f).name}")

    # gh release upload --clobber overwrites existing assets
    _run_gh([
        "release", "upload", RELEASE_TAG,
        "--clobber",
    ] + files_to_upload)

    if verbose:
        print("Upload complete.")


def download_artifacts(verbose: bool = True):
    """
    Download model files and SMA cache from GitHub Release.
    Files are saved to MODEL_DIR and SMA_CACHE_DIR respectively.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SMA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # List assets in the release
    result = _run_gh([
        "release", "view", RELEASE_TAG,
        "--json", "assets",
    ], check=False)

    if result.returncode != 0:
        raise RuntimeError(f"Release '{RELEASE_TAG}' not found. Run training first.")

    assets = json.loads(result.stdout).get("assets", [])

    if not assets:
        raise RuntimeError(f"No assets found in release '{RELEASE_TAG}'.")

    asset_names = [a["name"] for a in assets]
    if verbose:
        print(f"Found {len(asset_names)} assets in release '{RELEASE_TAG}':")
        for name in asset_names:
            print(f"  {name}")

    # Download all assets to a temp dir, then move to correct locations
    with tempfile.TemporaryDirectory() as tmpdir:
        _run_gh([
            "release", "download", RELEASE_TAG,
            "--dir", tmpdir,
        ])

        for fname in os.listdir(tmpdir):
            src = os.path.join(tmpdir, fname)

            if fname.endswith(".zip"):
                dst = SMA_CACHE_DIR / fname
            else:
                dst = MODEL_DIR / fname

            shutil.copy2(src, dst)
            if verbose:
                print(f"  Saved: {dst}")

    if verbose:
        print("Download complete.")


def main():
    """CLI entry point for storage operations."""
    import argparse
    parser = argparse.ArgumentParser(description="GitHub Release artifact storage")
    parser.add_argument("action", choices=["upload", "download"],
                        help="upload or download artifacts")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    args = parser.parse_args()

    if args.action == "upload":
        upload_artifacts(verbose=not args.quiet)
    else:
        download_artifacts(verbose=not args.quiet)


if __name__ == "__main__":
    main()

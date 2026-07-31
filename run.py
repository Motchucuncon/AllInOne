#!/usr/bin/env python3
"""One-click runner for the Faceless Review Video Generator.

Usage:
    python run.py

This script will:
1. Auto-install all dependencies from requirements.txt
2. Launch the Gradio web UI
3. Provide a public HTTPS link (share=True)
"""

import os
import subprocess
import sys


def main():
    repo_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("🎬 FACELESS REVIEW VIDEO GENERATOR")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Install dependencies
    # ------------------------------------------------------------------
    req_file = os.path.join(repo_dir, "requirements.txt")
    if os.path.exists(req_file):
        print("\n[1/2] 📦 Installing dependencies from requirements.txt...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "-r", req_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print("   ✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("   ⚠️  pip install failed, trying again with verbose output...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-r", req_file],
            )
    else:
        print("\n[1/2] ⚠️  requirements.txt not found, skipping auto-install.")
        print("   Make sure you have installed: gradio, requests, diffusers, torch, etc.")

    # ------------------------------------------------------------------
    # Step 2: Launch Gradio UI
    # ------------------------------------------------------------------
    print("\n[2/2] 🚀 Launching Gradio Web UI...")
    print("   ⏳ Please wait for the public URL to appear...")
    print("=" * 60)

    app_path = os.path.join(repo_dir, "app.py")
    if not os.path.exists(app_path):
        print(f"❌ ERROR: {app_path} not found!")
        sys.exit(1)

    # Run app.py
    subprocess.check_call([sys.executable, app_path])


if __name__ == "__main__":
    main()
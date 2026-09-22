#!/usr/bin/env python3
"""
Minab Web Dashboard Local Development Launcher
Starts Flask server and background vision worker.
"""

import sys
import os

# Ensure current directory is in sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

# Check if running inside project .venv, otherwise re-exec with .venv Python if available
venv_win = os.path.join(root_dir, ".venv", "Scripts", "python.exe")
venv_posix = os.path.join(root_dir, ".venv", "bin", "python")
target_venv_python = venv_win if os.path.exists(venv_win) else (venv_posix if os.path.exists(venv_posix) else None)

if target_venv_python and os.path.abspath(sys.executable).lower() != os.path.abspath(target_venv_python).lower() and "MINAB_VENV_REEXEC" not in os.environ:
    os.environ["MINAB_VENV_REEXEC"] = "1"
    print(f"[Minab] Switching environment to project .venv Python: {target_venv_python}")
    os.execv(target_venv_python, [target_venv_python] + sys.argv)

from backend.app import create_app

if __name__ == "__main__":
    app = create_app(start_worker=True)
    app.run(host="0.0.0.0", port=5000, debug=False)

#!/usr/bin/env python3
"""Launch InfraExplorer as a local desktop program."""

from __future__ import annotations

import os
import subprocess
import sys


ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def ensure_packages() -> None:
    needed = [("numpy", "numpy"), ("soundfile", "soundfile"), ("miniaudio", "miniaudio")]
    missing: list[str] = []
    for mod, pip_name in needed:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pip_name)
    if not missing:
        return
    print("Installing " + ", ".join(missing) + " (one-time)…")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing, "--quiet"])


def main() -> int:
    if "--selftest" in sys.argv:
        ensure_packages()
        from infraexplorer.engine import run_self_test

        failed = 0
        for case in run_self_test():
            mark = "PASS" if case["pass"] else "FAIL"
            print(f"  [{mark}] {case['name']}  —  {case['detail']}")
            if not case["pass"]:
                failed += 1
        return 1 if failed else 0

    ensure_packages()
    try:
        import tkinter  # noqa: F401
    except ImportError:
        print(
            "InfraExplorer needs Python with Tk (the desktop UI).\n"
            "On Windows, install Python from https://www.python.org/downloads/\n"
            "and keep the 'tcl/tk and IDLE' option checked.\n"
        )
        return 1

    from infraexplorer.app import main as run_app

    run_app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

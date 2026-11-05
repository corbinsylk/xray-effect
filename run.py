#!/usr/bin/env python3
"""One command: python3 run.py my-clip.mp4 --start 0
Sets everything up inside this folder the first time (no admin rights, nothing outside it), then makes the effect.
Works with any Python 3: if yours is too new or too old for the tracking library, it fetches its own copy of Python 3.12."""
import hashlib, os, shutil, subprocess, sys, urllib.request, venv
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV, TOOLS, MODELS = HERE / ".venv", HERE / ".tools", HERE / "models"
GOOD = {(3, 10), (3, 11), (3, 12)}
BASE = "https://storage.googleapis.com/mediapipe-models/"
FILES = {  # free MediaPipe models from Google (Apache 2.0): name -> (path, sha256)
    "pose_landmarker_heavy.task": ("pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task", "64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b"),
    "face_landmarker.task": ("face_landmarker/face_landmarker/float16/1/face_landmarker.task", "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"),
    "hand_landmarker.task": ("hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"),
    "selfie_multiclass_256x256.tflite": ("image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite", "c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0"),
}


def venv_python():
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(*cmd, **kw):
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


def find_python():
    if tuple(sys.version_info[:2]) in GOOD:
        return sys.executable
    for v in ("3.12", "3.11", "3.10"):
        p = shutil.which(f"python{v}")
        if p:
            return p
    return None


def make_venv():
    py = find_python()
    if py:
        run(py, "-m", "venv", VENV)
        return
    print("Getting Python 3.12 (stays in this folder)...", flush=True)
    env = dict(os.environ, UV_PYTHON_INSTALL_DIR=str(TOOLS / "python"), UV_CACHE_DIR=str(TOOLS / "cache"), PYTHONPATH=str(TOOLS / "uv"))
    run(sys.executable, "-m", "pip", "install", "--quiet", "--target", TOOLS / "uv", "uv==0.13.7")
    run(sys.executable, "-m", "uv", "venv", "--seed", "--python", "3.12", "--python-preference", "only-managed", VENV, env=env)


def setup():
    if not (VENV / ".ready").exists():
        print("First run: setting up (about 600 MB, a few minutes, only once)...", flush=True)
        if VENV.exists():
            shutil.rmtree(VENV)
        make_venv()
        run(venv_python(), "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "-r", HERE / "requirements.txt")
        (VENV / ".ready").write_text("ok")
    MODELS.mkdir(exist_ok=True)
    for name, (path, sha) in FILES.items():
        f = MODELS / name
        if not f.exists():
            print(f"Downloading {name}...", flush=True)
            tmp = f.with_suffix(".part")
            with urllib.request.urlopen(BASE + path, timeout=120) as r, open(tmp, "wb") as w:
                shutil.copyfileobj(r, w)
            if hashlib.sha256(tmp.read_bytes()).hexdigest() != sha:
                tmp.unlink()
                sys.exit(f"{name} did not download correctly. Run this again.")
            tmp.replace(f)


if __name__ == "__main__":
    setup()
    # keeps the tracking library's own log lines out of the way
    env = dict(os.environ, GLOG_minloglevel="2", TF_CPP_MIN_LOG_LEVEL="2")
    sys.exit(subprocess.run([str(venv_python()), str(HERE / "xray.py"), *sys.argv[1:]], env=env).returncode)

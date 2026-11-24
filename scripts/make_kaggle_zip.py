"""Pack the code (no .env, .venv, outputs) into dist/egig_code.zip for a Kaggle Dataset upload.

    python scripts/make_kaggle_zip.py
"""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["egig", "eval", "tests", "kaggle", "scripts", "README.md", "pyproject.toml", ".env.example",
           "requirements.txt", "requirements-vision.txt", "requirements-gpu.txt"]


def main() -> None:
    out = ROOT / "dist" / "egig_code.zip"
    out.parent.mkdir(exist_ok=True)
    count = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for item in INCLUDE:
            path = ROOT / item
            files = [path] if path.is_file() else sorted(path.rglob("*"))
            for f in files:
                if f.is_file() and "__pycache__" not in f.parts and f.name != ".env":
                    z.write(f, f.relative_to(ROOT).as_posix())
                    count += 1
    print(f"wrote {out} ({count} files, {out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

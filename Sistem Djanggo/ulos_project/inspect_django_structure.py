#!/usr/bin/env python3
import os
import re
import sys
from pathlib import Path

IGNORE_DIRS = {
    ".git", ".idea", ".vscode",
    "__pycache__", ".pytest_cache", ".mypy_cache",
    "venv", ".venv", "env", ".env",
    "node_modules",
    "static", "media",
    "dist", "build",
}
IGNORE_FILES_SUFFIX = {".pyc", ".pyo", ".log", ".sqlite3-journal"}
MAX_DEPTH = 5

def should_ignore_dir(p: Path) -> bool:
    return p.name in IGNORE_DIRS or p.name.startswith(".")

def should_ignore_file(p: Path) -> bool:
    return p.suffix in IGNORE_FILES_SUFFIX

def print_tree(root: Path, max_depth: int = MAX_DEPTH) -> None:
    root = root.resolve()
    print("\n=== project tree (max depth: {}) ===".format(max_depth))
    print(str(root))

    def walk(dir_path: Path, prefix: str, depth: int):
        if depth > max_depth:
            return

        try:
            items = sorted(dir_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except PermissionError:
            print(prefix + "└── [permission denied]")
            return

        items = [x for x in items if not should_ignore_dir(x) and not should_ignore_file(x)]
        for i, item in enumerate(items):
            last = (i == len(items) - 1)
            connector = "└── " if last else "├── "
            print(prefix + connector + item.name)

            if item.is_dir():
                extension = "    " if last else "│   "
                walk(item, prefix + extension, depth + 1)

    walk(root, "", 1)

def find_manage_py(root: Path) -> Path | None:
    candidates = list(root.glob("manage.py"))
    if candidates:
        return candidates[0].resolve()
    # cari lebih dalam (maks 4 level) kalau manage.py tidak ada di root saat ini
    for p in root.rglob("manage.py"):
        parts = p.relative_to(root).parts
        if len(parts) <= 4 and not any(part in IGNORE_DIRS for part in parts):
            return p.resolve()
    return None

def extract_settings_module(manage_py: Path) -> str | None:
    try:
        text = manage_py.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    m = re.search(r"DJANGO_SETTINGS_MODULE\s*=\s*['\"]([^'\"]+)['\"]", text)
    if m:
        return m.group(1).strip()
    return None

def guess_project_package(manage_py: Path) -> Path | None:
    # biasanya: manage.py ada di root project, package ada di folder sibling (mis: config/ atau projectname/)
    base = manage_py.parent
    # cari folder yang punya settings.py / urls.py
    for d in sorted(base.iterdir()):
        if d.is_dir() and not should_ignore_dir(d):
            if (d / "settings.py").exists() or (d / "urls.py").exists():
                return d.resolve()
            # juga handle struktur settings package: settings/base.py
            if (d / "settings").is_dir():
                return d.resolve()
    return None

def find_key_files(root: Path):
    patterns = [
        "**/settings.py",
        "**/urls.py",
        "**/views.py",
        "**/apps.py",
        "**/wsgi.py",
        "**/asgi.py",
        "**/requirements.txt",
        "**/pyproject.toml",
    ]
    found = {}
    for pat in patterns:
        found[pat] = []
        for p in root.rglob(pat.replace("**/", "")):
            rel = p.relative_to(root)
            if any(part in IGNORE_DIRS for part in rel.parts):
                continue
            if should_ignore_file(p):
                continue
            # batasi hasil biar tidak kebanyakan
            if len(rel.parts) > 8:
                continue
            found[pat].append(str(rel))
    return found

def main():
    cwd = Path.cwd().resolve()

    print("=== django project inspector ===")
    print("cwd:", cwd)

    manage_py = find_manage_py(cwd)
    if not manage_py:
        print("\n[!] manage.py tidak ketemu di folder ini.")
        print("coba cd ke folder yang ada manage.py, lalu jalankan lagi.")
        sys.exit(1)

    print("\n=== manage.py found ===")
    print(manage_py)

    settings_module = extract_settings_module(manage_py)
    env_settings = os.environ.get("DJANGO_SETTINGS_MODULE")

    print("\n=== settings module detection ===")
    print("from manage.py:", settings_module or "(tidak ketemu di manage.py)")
    print("from env var DJANGO_SETTINGS_MODULE:", env_settings or "(tidak diset)")

    pkg_guess = guess_project_package(manage_py)
    if pkg_guess:
        print("\n=== likely django 'project package' folder ===")
        print(pkg_guess)
    else:
        print("\n=== likely django 'project package' folder ===")
        print("(belum bisa ditebak otomatis)")

    # cetak tree dari folder tempat manage.py berada (root project yang paling relevan)
    project_root = manage_py.parent.resolve()
    print_tree(project_root, MAX_DEPTH)

    print("\n=== key files (guesses) ===")
    key = find_key_files(project_root)
    for pat, items in key.items():
        if items:
            print(f"\n{pat}:")
            for it in sorted(items)[:25]:
                print(" -", it)

    print("\n=== what to paste here ===")
    print("1) output lengkap script ini")
    print("2) path mana yang berisi settings.py dan urls.py (kalau terlihat di output)")
    print("\ncatatan keamanan:")
    print("- output ini hanya list nama file/folder. tidak menampilkan isi SECRET_KEY/password.")
    print("- kalau kamu punya folder sensitif, bilang aja, nanti aku bantu filter.")

if __name__ == "__main__":
    main()
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _prepare_local_jupyter_env(project_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    venv_python = Path(sys.executable)
    venv_bin = str(venv_python.parent)
    paths = {
        "JUPYTER_CONFIG_DIR": project_root / ".jupyter",
        "JUPYTER_DATA_DIR": project_root / ".local" / "share" / "jupyter",
        "JUPYTER_RUNTIME_DIR": project_root / ".local" / "share" / "jupyter" / "runtime",
        "IPYTHONDIR": project_root / ".ipython",
        "XDG_CACHE_HOME": project_root / ".cache",
        "MPLCONFIGDIR": project_root / ".cache" / "matplotlib",
    }
    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
    env["MUSIC_CATEGORIZER_PROJECT_ROOT"] = str(project_root)
    return env


def _write_kernel_spec(kernels_dir: Path, display_name: str, project_root: Path) -> Path:
    kernels_dir.mkdir(parents=True, exist_ok=True)
    kernel_json = kernels_dir / "kernel.json"
    spec = {
        "argv": [
            str(Path(sys.executable)),
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ],
        "display_name": display_name,
        "language": "python",
        "metadata": {
            "project_root": str(project_root),
        },
    }
    kernel_json.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return kernels_dir


def _ensure_project_kernel(project_root: Path, env: dict[str, str]) -> Path:
    kernels_root = Path(env["JUPYTER_DATA_DIR"]) / "kernels"
    _write_kernel_spec(kernels_root / "python3", "Python 3 (.venv)", project_root)
    return _write_kernel_spec(kernels_root / "music-categorizer", "Music Categorizer (.venv)", project_root)


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    notebook_path = project_root / "music_scale_lab.ipynb"
    env = _prepare_local_jupyter_env(project_root)
    _ensure_project_kernel(project_root, env)
    extra_args = sys.argv[1:]
    subprocess.run(
        [sys.executable, "-m", "notebook", str(notebook_path), *extra_args],
        cwd=project_root,
        env=env,
        check=False,
    )


if __name__ == "__main__":
    main()
